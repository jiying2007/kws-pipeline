#!/usr/bin/python3
"""Ephemeral own-repository credential handoff, with no persistent credential file."""
import os
import sys

if len(sys.argv) != 2 or os.environ.get('KWS_GIT_REPOSITORY') not in ('jiying2007/kws-pipeline', 'jiying2007/kws-data'):
    sys.exit(1)
repo = os.environ['KWS_GIT_REPOSITORY']
prompts = {
    "Username for 'https://github.com/" + repo + ".git': ": 'x-access-token',
    "Password for 'https://x-access-token@github.com/" + repo + ".git': ": os.environ.get('KWS_GIT_TOKEN', ''),
}
answer = prompts.get(sys.argv[1])
if not answer or '\n' in answer or '\r' in answer:
    sys.exit(1)
sys.stdout.write(answer + '\n')
