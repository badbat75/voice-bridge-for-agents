#!/usr/bin/env bash
# Install (or uninstall) the openclaw-voicebridge user systemd unit
# and the matching udev permission rule.
#
# The unit is a USER unit (lives under ~/.config/systemd/user) — it
# runs as the invoking user, no `User=` directive. The udev rule
# only sets MODE/GROUP on /dev/hidraw* for the Jabra so the user can
# open it; the bridge handles plug/unplug internally via its
# reconnect loop, so udev does NOT trigger the service.
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
USER_UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
SERVICE_DST="$USER_UNIT_DIR/$SERVICE_NAME"
RULES_DST="/etc/udev/rules.d/$RULES_NAME"
LEGACY_SERVICE_DST="/etc/systemd/system/$SERVICE_NAME"

die() { echo "ERROR: $*" >&2; exit 1; }
say() { printf '  %s\n' "$*"; }

ensure_user_session_env() {
    # `systemctl --user` talks to the per-user systemd manager over a
    # D-Bus socket under $XDG_RUNTIME_DIR. PAM sets these for login
    # shells, but plain `su -`, `sudo -u`, and some non-interactive
    # SSH sessions don't — so the script bails with
    # "Failed to connect to user scope bus". Linger guarantees the
    # socket exists at /run/user/$UID/bus, so we can fix it up.
    local uid
    uid="$(id -u)"
    if [ -z "${XDG_RUNTIME_DIR:-}" ]; then
        export XDG_RUNTIME_DIR="/run/user/$uid"
    fi
    if [ -z "${DBUS_SESSION_BUS_ADDRESS:-}" ]; then
        export DBUS_SESSION_BUS_ADDRESS="unix:path=$XDG_RUNTIME_DIR/bus"
    fi
    if ! [ -S "$XDG_RUNTIME_DIR/bus" ]; then
        die "$XDG_RUNTIME_DIR/bus is not a socket — user systemd manager is not running for $USER. Enable linger and start it: 'sudo loginctl enable-linger $USER && sudo systemctl start user@$uid.service'"
    fi
}

cleanup_legacy_system_install() {
    # Earlier versions of this script linked the unit under
    # /etc/systemd/system as a system unit. Detect that install and
    # clean it up so the new user unit isn't shadowed.
    if [ -e "$LEGACY_SERVICE_DST" ] || [ -L "$LEGACY_SERVICE_DST" ]; then
        echo "Removing legacy system install of $SERVICE_NAME"
        sudo systemctl stop "$SERVICE_NAME" 2>/dev/null || true
        sudo systemctl disable "$SERVICE_NAME" 2>/dev/null || true
        sudo rm -f "$LEGACY_SERVICE_DST"
        sudo systemctl daemon-reload
        say "removed $LEGACY_SERVICE_DST"
    fi
}

uninstall() {
    echo "Uninstalling $SERVICE_NAME (user)"
    systemctl --user stop "$SERVICE_NAME" 2>/dev/null || true
    systemctl --user disable "$SERVICE_NAME" 2>/dev/null || true
    rm -f "$SERVICE_DST"
    sudo rm -f "$RULES_DST"
    systemctl --user daemon-reload
    sudo udevadm control --reload-rules
    say "removed $SERVICE_DST"
    say "removed $RULES_DST"
    cleanup_legacy_system_install
    say "linger left enabled — 'sudo loginctl disable-linger $USER' to drop it"
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
    say "linger: $(loginctl show-user "$USER" --property=Linger --value 2>/dev/null || echo unknown)"
    if [ -e "$LEGACY_SERVICE_DST" ] || [ -L "$LEGACY_SERVICE_DST" ]; then
        say "WARNING: legacy system unit still present at $LEGACY_SERVICE_DST"
    fi
    echo
    echo "=== systemctl --user status ==="
    systemctl --user status "$SERVICE_NAME" --no-pager 2>&1 || true
    echo
    echo "=== last 20 log lines ==="
    journalctl --user -u "$SERVICE_NAME" -n 20 --no-pager 2>&1 || true
}

ensure_linger() {
    if [ "$(loginctl show-user "$USER" --property=Linger --value 2>/dev/null)" = "yes" ]; then
        return
    fi
    echo "Enabling linger so the user unit starts at boot without a login session"
    sudo loginctl enable-linger "$USER"
}

check_groups() {
    # The user must be in `plugdev` (for /dev/hidraw* via the udev rule)
    # and `audio` (for ALSA). Without these the unit starts but the
    # bridge fails at first device open or first aplay call.
    local missing=()
    for grp in plugdev audio; do
        if ! id -nG "$USER" | tr ' ' '\n' | grep -qx "$grp"; then
            missing+=("$grp")
        fi
    done
    if [ ${#missing[@]} -gt 0 ]; then
        echo "WARNING: $USER is not in: ${missing[*]}"
        echo "  Run: sudo usermod -aG ${missing[*]// /,} $USER"
        echo "  Then log out and back in (or reboot) for groups to take effect."
        echo
    fi
}

install_unit() {
    [ -f "$SERVICE_SRC" ] || die "missing $SERVICE_SRC"
    [ -f "$RULES_SRC" ] || die "missing $RULES_SRC"

    # Sanity: the unit's WorkingDirectory should point at *this* tree.
    # If someone copied this directory elsewhere without editing the
    # service file, starting the unit would launch the wrong instance.
    if ! grep -qx "WorkingDirectory=$WORKSPACE" "$SERVICE_SRC"; then
        echo "WARNING: $SERVICE_NAME has a WorkingDirectory other than this tree."
        say "this script lives under: $WORKSPACE"
        grep "^WorkingDirectory" "$SERVICE_SRC" | sed 's/^/  unit says:        /'
        echo "  Edit $SERVICE_SRC if the unit should run from $WORKSPACE."
        echo
    fi

    check_groups
    cleanup_legacy_system_install
    ensure_linger

    echo "Linking $SERVICE_NAME via systemctl --user link"
    mkdir -p "$USER_UNIT_DIR"
    systemctl --user link "$SERVICE_SRC"

    echo "Linking $RULES_NAME"
    sudo ln -sf "$RULES_SRC" "$RULES_DST"

    echo "Reloading user systemd + udev"
    systemctl --user daemon-reload
    sudo udevadm control --reload-rules

    echo "Re-applying udev rules to currently-attached hidraw devices"
    # MODE/GROUP is set on the kernel `add` event. Re-emitting `add`
    # for already-plugged devices makes the new permissions take
    # effect without an unplug/replug cycle.
    sudo udevadm trigger --subsystem-match=hidraw --action=add

    echo "Enabling and starting the unit"
    systemctl --user enable --now "$SERVICE_NAME"

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
