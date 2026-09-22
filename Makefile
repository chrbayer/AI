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
BASHCOMP  = $(DESTDIR)$(PREFIX)/share/bash-completion/completions
ZSHCOMP   = $(DESTDIR)$(PREFIX)/share/zsh/site-functions

.PHONY: install install-link uninstall test

# Everything that answers without a GPU, a model or the network.
test:
	shellcheck -S warning llmctl patches/build-tts-server.sh tests/smoke.sh completions/llmctl.bash
	tests/smoke.sh
	python3 tests/test_python.py

install:
	install -Dm755 llmctl $(BINDIR)/llmctl
	install -Dm644 proxy.py $(LIBDIR)/proxy.py
	install -Dm644 anthropic_compat.py $(LIBDIR)/anthropic_compat.py
	install -Dm644 halogen_bench.py $(LIBDIR)/halogen_bench.py
	install -Dm644 comfyui_models.py $(LIBDIR)/comfyui_models.py
	install -Dm644 tts_server.py $(LIBDIR)/tts_server.py
	install -Dm644 images_server.py $(LIBDIR)/images_server.py
	install -Dm644 tts_voice_design.py $(LIBDIR)/tts_voice_design.py
	install -Dm644 -t $(SHAREDIR)/templates templates/*
	install -Dm644 -t $(SHAREDIR)/examples examples/*.conf
	install -Dm644 -t $(SHAREDIR)/comfyui/workflows comfyui/workflows/*.json
	install -Dm644 -t $(SHAREDIR)/comfyui/api comfyui/api/*.json
	install -Dm644 -t $(SHAREDIR)/comfyui/patches comfyui/patches/*.patch
	install -Dm644 comfyui/sources.json $(SHAREDIR)/comfyui/sources.json
	install -Dm644 comfyui/custom_nodes.txt $(SHAREDIR)/comfyui/custom_nodes.txt
	install -Dm644 -t $(SHAREDIR)/tts/voices tts/voices/*
	install -Dm755 patches/build-tts-server.sh $(SHAREDIR)/patches/build-tts-server.sh
	install -Dm644 -t $(SHAREDIR)/patches patches/llama.cpp-pr26603-*.patch
	install -Dm644 README.md $(DOCDIR)/README.md
	install -Dm644 completions/llmctl.bash $(BASHCOMP)/llmctl
	install -Dm644 completions/_llmctl $(ZSHCOMP)/_llmctl

install-link:
	install -d $(BINDIR) $(BASHCOMP) $(ZSHCOMP)
	ln -sfn $(CURDIR)/llmctl $(BINDIR)/llmctl
	ln -sfn $(CURDIR)/completions/llmctl.bash $(BASHCOMP)/llmctl
	ln -sfn $(CURDIR)/completions/_llmctl $(ZSHCOMP)/_llmctl

uninstall:
	rm -f $(BINDIR)/llmctl $(BASHCOMP)/llmctl $(ZSHCOMP)/_llmctl
	rm -rf $(LIBDIR) $(SHAREDIR) $(DOCDIR)
