"""Authenticated task session with an optional loopback-only VPN proxy tunnel.

Credentials come only from a named process environment variable. Commands are
read from local files; uploads are not used to deploy experimental source code.
Use Git fetch/pull through the proxy for production deployments.
"""
import argparse
import json
import logging
import os
from pathlib import Path
import select
import socket
import sys
import threading
import time

import paramiko


def bridge(channel, proxy_host, proxy_port):
    upstream = None
    try:
        upstream = socket.create_connection((proxy_host, proxy_port), timeout=20)
        while True:
            ready, _, _ = select.select([channel, upstream], [], [], 30)
            for source in ready:
                data = source.recv(65536)
                if not data:
                    return
                (upstream if source is channel else channel).sendall(data)
    finally:
        channel.close()
        if upstream is not None:
            upstream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', required=True)
    parser.add_argument('--port', type=int, default=22)
    parser.add_argument('--user', default='root')
    parser.add_argument('--password-env', default='MOLSTEER_SCNET_PASSWORD')
    parser.add_argument('--local-proxy-port', type=int)
    parser.add_argument('--remote-proxy-port', type=int, default=17897)
    args = parser.parse_args()
    logging.getLogger('paramiko').setLevel(logging.CRITICAL)
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(args.host, port=args.port, username=args.user,
                   password=os.environ[args.password_env], timeout=20,
                   auth_timeout=25, banner_timeout=20, look_for_keys=False, allow_agent=False)
    transport = client.get_transport()
    transport.set_keepalive(30)
    try:
        if args.local_proxy_port:
            def forwarded(channel, origin, server):
                threading.Thread(target=bridge, args=(channel, '127.0.0.1', args.local_proxy_port), daemon=True).start()
            bound = transport.request_port_forward('127.0.0.1', args.remote_proxy_port, handler=forwarded)
            print(json.dumps({'remote_proxy': f'http://127.0.0.1:{bound}',
                              'local_vpn_proxy': f'http://127.0.0.1:{args.local_proxy_port}'}), flush=True)
        print('CONNECTED_READY', flush=True)
        for line in sys.stdin:
            request = json.loads(line)
            if request.get('exit'):
                break
            if request.get('uploads') or request.get('downloads'):
                with client.open_sftp() as sftp:
                    for local, remote in request.get('uploads', []):
                        sftp.put(local, remote)
                        print('UPLOADED', remote, flush=True)
                    for remote, local in request.get('downloads', []):
                        Path(local).parent.mkdir(parents=True, exist_ok=True)
                        sftp.get(remote, local)
                        print('DOWNLOADED', local, flush=True)
            if request.get('command_file'):
                channel = transport.open_session()
                channel.exec_command(Path(request['command_file']).read_text(encoding='utf-8'))
                while not channel.exit_status_ready() or channel.recv_ready() or channel.recv_stderr_ready():
                    if channel.recv_ready():
                        sys.stdout.buffer.write(channel.recv(65536)); sys.stdout.buffer.flush()
                    if channel.recv_stderr_ready():
                        sys.stderr.buffer.write(channel.recv_stderr(65536)); sys.stderr.buffer.flush()
                    time.sleep(.05)
                print('REMOTE_EXIT', channel.recv_exit_status(), flush=True)
                channel.close()
            print('COMMAND_DONE', flush=True)
    finally:
        client.close()


if __name__ == '__main__':
    main()
