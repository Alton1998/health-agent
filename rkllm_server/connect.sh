#!/bin/bash
# connect_wifi.sh

SSID="UofA2023"
PASSWORD="SneTom220800"

# Enable Wi-Fi device (usually wlan0)
nmcli radio wifi on

# Connect to the Wi-Fi network
nmcli dev wifi connect "$SSID" password "$PASSWORD"

# Optional: make connection autoconnect on boot
nmcli connection modify "$SSID" connection.autoconnect yes
