#!/usr/bin/env bash
# Install (or uninstall) the openclaw-voicebridge systemd unit + udev rule.
#
# Links — does NOT copy — both files into /etc/ so that edits to the
# workspace versions take effect after a daemon-reload / rules-reload.
#
# Usage:
#   ./install.sh           install (default)
#   ./install.sh -u        uninstall
#   ./install.sh --status  show current install state and recent logs
#
# Run from anywhere — the script self-locates relative to its own path.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(dirname "$HERE")"

SERVICE_NAME="openclaw-voicebridge.service"
RULES_NAME="99-openclaw-voicebridge.rules"
SERVICE_SRC="$HERE/$SERVICE_NAME"
RULES_SRC="$HERE/$RULES_NAME"
SERVICE_DST="/etc/systemd/system/$SERVICE_NAME"
RULES_DST="/etc/udev/rules.d/$RULES_NAME"

die() { echo "ERROR: $*" >&2; exit 1; }
say() { printf '  %s\n' "$*"; }

uninstall() {
    echo "Uninstalling $SERVICE_NAME"
    sudo systemctl stop "$SERVICE_NAME" 2>/dev/null || true
    sudo systemctl disable "$SERVICE_NAME" 2>/dev/null || true
    sudo rm -f "$SERVICE_DST"
    sudo rm -f "$RULES_DST"
    sudo systemctl daemon-reload
    sudo udevadm control --reload-rules
    say "removed $SERVICE_DST"
    say "removed $RULES_DST"
    say "reloaded systemd + udev"
    echo "Done."
}

show_status() {
    echo "=== install state ==="
    for path in "$SERVICE_DST" "$RULES_DST"; do
        if [ -L "$path" ]; then
            target="$(readlink -f "$path")"
            say "$path → $target"
        elif [ -e "$path" ]; then
            say "$path (regular file, not a symlink)"
        else
            say "$path (absent)"
        fi
    done
    echo
    echo "=== systemctl status ==="
    systemctl status "$SERVICE_NAME" --no-pager 2>&1 || true
    echo
    echo "=== last 20 log lines ==="
    journalctl -u "$SERVICE_NAME" -n 20 --no-pager 2>&1 || true
}

install_unit() {
    [ -f "$SERVICE_SRC" ] || die "missing $SERVICE_SRC"
    [ -f "$RULES_SRC" ] || die "missing $RULES_SRC"

    # Sanity-check that the unit's WorkingDirectory points to *this*
    # workspace. If someone copied this tree elsewhere without editing the
    # service file, starting the unit would launch the wrong instance.
    if ! grep -qx "WorkingDirectory=$WORKSPACE" "$SERVICE_SRC"; then
        echo "WARNING: $SERVICE_NAME has a WorkingDirectory other than this tree."
        say "this script lives under: $WORKSPACE"
        grep "^WorkingDirectory" "$SERVICE_SRC" | sed 's/^/  unit says:        /'
        echo "  Edit $SERVICE_SRC if the unit should run from $WORKSPACE."
        echo
    fi

    echo "Linking $SERVICE_NAME via systemctl link"
    sudo systemctl link "$SERVICE_SRC"

    echo "Linking $RULES_NAME"
    sudo ln -sf "$RULES_SRC" "$RULES_DST"

    echo "Reloading systemd + udev"
    sudo systemctl daemon-reload
    sudo udevadm control --reload-rules

    echo "Triggering hidraw add-events for already-plugged devices"
    # --action=add is required: the rule has ACTION=="add" so a default
    # "change" trigger wouldn't fire SYSTEMD_WANTS.
    sudo udevadm trigger --subsystem-match=hidraw --action=add

    echo
    show_status
}

case "${1:-install}" in
    install|"")           install_unit ;;
    -u|--uninstall)       uninstall ;;
    -s|--status)          show_status ;;
    -h|--help)
        sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
        ;;
    *)
        die "unknown option: $1 (try --help)"
        ;;
esac
