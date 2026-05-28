from ldap3 import Server, Connection, ALL, SIMPLE
import auth_ad
from flask import Flask, render_template, request, redirect, url_for, session, flash

# Configuración del servidor Windows Server
LDAP_SERVER = '192.168.1.100'  # Cambia por la IP de tu Windows Server / Domain Controller
DOMAIN_SUFFIX = '@lab.local'  # Active Directory domain suffix

def authenticate_ad_user(username, password):
    # --- LOCAL TEST MODE (for when the Windows Server is offline) ---
    if username == 'admin.test' and password == 'Machine123':
        print("Access granted using local test user.")
        return True
    # -----------------------------------------------------------------

    # Real connection logic to Windows Server
    if not username.endswith(DOMAIN_SUFFIX):
        user_dn = f"{username}{DOMAIN_SUFFIX}"
    else:
        user_dn = username

    try:
        server = Server(LDAP_SERVER, get_info=ALL)
        conn = Connection(server, user=user_dn, password=password, authentication=SIMPLE)
        if conn.bind():
            conn.unbind()
            return True
        return False
    except Exception as e:
        print(f"Error connecting to Windows Server: {e}")
        return False