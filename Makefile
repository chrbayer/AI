# Install llmctl. The program goes under $(PREFIX); everything a user owns —
# model definitions, presets, models, logs — stays in their home directory
# (see README, "Installation & directories").
#
#   sudo make install         copy the program to /usr/local
#   sudo make install-link    link /usr/local/bin/llmctl to this checkout instead,
#                             so edits here take effect without reinstalling
#   sudo make uninstall
#
# `install` writes each file as a new inode, so a command still running from the
# old llmctl (a long `preset`, say) is not pulled out from under itself.

PREFIX   ?= /usr/local
BINDIR    = $(DESTDIR)$(PREFIX)/bin
LIBDIR    = $(DESTDIR)$(PREFIX)/lib/llmctl
SHAREDIR  = $(DESTDIR)$(PREFIX)/share/llmctl
DOCDIR    = $(DESTDIR)$(PREFIX)/share/doc/llmctl

.PHONY: install install-link uninstall

install:
	install -Dm755 llmctl $(BINDIR)/llmctl
	install -Dm644 proxy.py $(LIBDIR)/proxy.py
	install -Dm644 anthropic_compat.py $(LIBDIR)/anthropic_compat.py
	install -Dm644 halogen_bench.py $(LIBDIR)/halogen_bench.py
	install -Dm644 comfyui_models.py $(LIBDIR)/comfyui_models.py
	install -Dm644 tts_server.py $(LIBDIR)/tts_server.py
	install -Dm644 tts_voice_design.py $(LIBDIR)/tts_voice_design.py
	install -Dm644 -t $(SHAREDIR)/templates templates/*
	install -Dm644 -t $(SHAREDIR)/examples examples/*.conf
	install -Dm644 -t $(SHAREDIR)/comfyui/workflows comfyui/workflows/*.json
	install -Dm644 -t $(SHAREDIR)/comfyui/patches comfyui/patches/*.patch
	install -Dm644 comfyui/sources.json $(SHAREDIR)/comfyui/sources.json
	install -Dm644 comfyui/custom_nodes.txt $(SHAREDIR)/comfyui/custom_nodes.txt
	install -Dm644 -t $(SHAREDIR)/tts/voices tts/voices/*
	install -Dm644 README.md $(DOCDIR)/README.md

install-link:
	install -d $(BINDIR)
	ln -sfn $(CURDIR)/llmctl $(BINDIR)/llmctl

uninstall:
	rm -f $(BINDIR)/llmctl
	rm -rf $(LIBDIR) $(SHAREDIR) $(DOCDIR)
