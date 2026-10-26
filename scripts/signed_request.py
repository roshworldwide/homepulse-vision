"""Call the Function URL with SigV4 (for AWS_IAM auth). Usage: python scripts/signed_request.py URL [METHOD] [JSON_BODY]"""
import json
import sys
import urllib.request

import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

url = sys.argv[1]
method = sys.argv[2] if len(sys.argv) > 2 else "GET"
body = sys.argv[3].encode() if len(sys.argv) > 3 else None
creds = boto3.Session().get_credentials().get_frozen_credentials()
req = AWSRequest(method=method, url=url, data=body, headers={"content-type": "application/json"} if body else {})
SigV4Auth(creds, "lambda", "us-east-1").add_auth(req)
r = urllib.request.Request(url, data=body, method=method, headers=dict(req.headers))
try:
    with urllib.request.urlopen(r, timeout=90) as resp:
        print(resp.status, resp.read().decode()[:4000])
except urllib.error.HTTPError as e:
    print(e.code, e.read().decode()[:2000])
