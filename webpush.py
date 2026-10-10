"""Web Push without a library of its own: the message encrypted for the
browser (RFC 8291, aes128gcm) and the sender signed with VAPID (RFC 8292),
both with `cryptography`, sent with `requests`.

    key = load_key(path)              # the server's P-256 key, made on first use
    public_key(key)                   # applicationServerKey for pushManager.subscribe
    send(subscription, b"...", key)   # → the push service's HTTP status
"""
import base64
import json
import os
import struct
import time
from pathlib import Path
from urllib.parse import urlsplit

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

# Apple refuses a VAPID subject that is no mailto: or https: URL.
SUBJECT = "https://github.com/chrbayer/AI"
RECORD_SIZE = 4096


def b64u(data):
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def unb64u(text):
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def load_key(path):
    """The server's VAPID key; made and kept (0600) the first time."""
    path = Path(path)
    if path.is_file():
        return serialization.load_pem_private_key(path.read_bytes(), password=None)
    key = ec.generate_private_key(ec.SECP256R1())
    pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                            serialization.NoEncryption())
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(pem)
    return key


def raw_public(key):
    return key.public_key().public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)


def public_key(key):
    return b64u(raw_public(key))


def hkdf(salt, info, length, ikm):
    return HKDF(algorithm=hashes.SHA256(), length=length, salt=salt, info=info).derive(ikm)


def encrypt(payload, p256dh, auth, salt=None, sender=None):
    """The body for the push service: header (salt, record size, our key) and
    one record, padded with the delimiter 0x02 (RFC 8188/8291)."""
    salt = salt or os.urandom(16)
    sender = sender or ec.generate_private_key(ec.SECP256R1())
    ua = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), p256dh)
    as_pub = raw_public(sender)
    shared = sender.exchange(ec.ECDH(), ua)
    ikm = hkdf(auth, b"WebPush: info\x00" + p256dh + as_pub, 32, shared)
    cek = hkdf(salt, b"Content-Encoding: aes128gcm\x00", 16, ikm)
    nonce = hkdf(salt, b"Content-Encoding: nonce\x00", 12, ikm)
    if len(payload) + 1 + 16 > RECORD_SIZE:
        raise ValueError("the message is too long for one record")
    record = AESGCM(cek).encrypt(nonce, payload + b"\x02", None)
    return salt + struct.pack(">IB", RECORD_SIZE, len(as_pub)) + as_pub + record


def vapid(endpoint, key, now=None):
    """The Authorization header: a JWT (ES256) for the push service's origin."""
    u = urlsplit(endpoint)
    claims = {"aud": f"{u.scheme}://{u.netloc}", "exp": int(now or time.time()) + 12 * 3600, "sub": SUBJECT}
    signing = (b64u(json.dumps({"typ": "JWT", "alg": "ES256"}, separators=(",", ":")).encode()) + "."
               + b64u(json.dumps(claims, separators=(",", ":")).encode()))
    r, s = decode_dss_signature(key.sign(signing.encode(), ec.ECDSA(hashes.SHA256())))
    token = signing + "." + b64u(r.to_bytes(32, "big") + s.to_bytes(32, "big"))
    return f"vapid t={token}, k={public_key(key)}"


def send(subscription, payload, key, ttl=6 * 3600, timeout=15):
    """Send one message; the status tells: 201 sent, 404/410 the subscription is gone."""
    k = subscription["keys"]
    body = encrypt(payload, unb64u(k["p256dh"]), unb64u(k["auth"]))
    r = requests.post(subscription["endpoint"], data=body, timeout=timeout, headers={
        "TTL": str(ttl), "Urgency": "normal", "Content-Encoding": "aes128gcm",
        "Content-Type": "application/octet-stream", "Authorization": vapid(subscription["endpoint"], key)})
    return r.status_code
