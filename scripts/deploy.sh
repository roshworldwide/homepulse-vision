#!/usr/bin/env bash
# Build the arm64 container, push to ECR, create/update the Lambda + Function URL.
# Idempotent. Usage:
#   scripts/deploy.sh                     # Function URL with AWS_IAM auth (private)
#   URL_AUTH=NONE scripts/deploy.sh       # public Function URL (rate limits are in the app)
#   BUDGET_EMAIL=you@example.com BUDGET_USD=10 scripts/deploy.sh   # also create the Budgets alarm
set -euo pipefail
cd "$(dirname "$0")/.."

REGION="${AWS_REGION:-us-east-1}"
NAME=homepulse-vision
ROLE=homepulse-vision-exec
URL_AUTH="${URL_AUTH:-AWS_IAM}"
RESERVED="${RESERVED_CONCURRENCY:-3}"
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REGISTRY="$ACCOUNT.dkr.ecr.$REGION.amazonaws.com"
TAG="$(git rev-parse --short HEAD)$([ -n "$(git status --porcelain)" ] && echo "-dirty-$(date +%s)" || true)"
IMAGE="$REGISTRY/$NAME:$TAG"

echo "== ECR repository"
aws ecr describe-repositories --region "$REGION" --repository-names "$NAME" >/dev/null 2>&1 ||
  aws ecr create-repository --region "$REGION" --repository-name "$NAME" \
    --image-scanning-configuration scanOnPush=true --image-tag-mutability IMMUTABLE >/dev/null
aws ecr put-lifecycle-policy --region "$REGION" --repository-name "$NAME" --lifecycle-policy-text \
  '{"rules":[{"rulePriority":1,"description":"keep last 5","selection":{"tagStatus":"any","countType":"imageCountMoreThan","countNumber":5},"action":{"type":"expire"}}]}' >/dev/null

echo "== build + push $IMAGE (linux/arm64)"
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$REGISTRY" >/dev/null
docker build --platform linux/arm64 --provenance=false -t "$IMAGE" .
docker push "$IMAGE" >/dev/null

echo "== execution role (Bedrock InvokeModel on Nova 2 Lite + own log group only)"
if ! aws iam get-role --role-name "$ROLE" >/dev/null 2>&1; then
  aws iam create-role --role-name "$ROLE" --assume-role-policy-document file://deploy/trust-policy.json \
    --description "HomePulse Vision Lambda: Bedrock Nova 2 Lite + CloudWatch Logs only" >/dev/null
  sleep 10
fi
POLICY=$(sed -e "s/\${REGION}/$REGION/g" -e "s/\${ACCOUNT}/$ACCOUNT/g" deploy/exec-role-policy.json)
aws iam put-role-policy --role-name "$ROLE" --policy-name least-privilege --policy-document "$POLICY"
ROLE_ARN="arn:aws:iam::$ACCOUNT:role/$ROLE"

echo "== demo issuer key (local .keys/, gitignored, never printed)"
mkdir -p .keys && chmod 700 .keys
if [ ! -f .keys/issuer_sk.b64 ]; then
  python3 -c "import os,base64;print(base64.b64encode(os.urandom(32)).decode())" > .keys/issuer_sk.b64
  chmod 600 .keys/issuer_sk.b64
fi
ENVFILE=$(mktemp); chmod 600 "$ENVFILE"; trap 'rm -f "$ENVFILE"' EXIT
python3 - "$ENVFILE" <<PY
import json, sys
sk = open(".keys/issuer_sk.b64").read().strip()
json.dump({"Variables": {"HPV_ISSUER_SK": sk, "HPV_TRACE_STDOUT": "1", "HPV_BEDROCK_REGION": "$REGION"}}, open(sys.argv[1], "w"))
PY

echo "== Lambda function"
if aws lambda get-function --region "$REGION" --function-name "$NAME" >/dev/null 2>&1; then
  aws lambda update-function-code --region "$REGION" --function-name "$NAME" --image-uri "$IMAGE" --architectures arm64 >/dev/null
  aws lambda wait function-updated-v2 --region "$REGION" --function-name "$NAME"
  aws lambda update-function-configuration --region "$REGION" --function-name "$NAME" \
    --environment "file://$ENVFILE" --memory-size 2048 --timeout 60 >/dev/null
else
  aws lambda create-function --region "$REGION" --function-name "$NAME" --package-type Image \
    --code ImageUri="$IMAGE" --role "$ROLE_ARN" --architectures arm64 --memory-size 2048 --timeout 60 \
    --environment "file://$ENVFILE" --description "HomePulse Vision: OpenCV 5 perception + Nova 2 Lite agent" >/dev/null
fi
aws lambda wait function-updated-v2 --region "$REGION" --function-name "$NAME"

echo "== logs: 14-day retention"
aws logs create-log-group --region "$REGION" --log-group-name "/aws/lambda/$NAME" 2>/dev/null || true
aws logs put-retention-policy --region "$REGION" --log-group-name "/aws/lambda/$NAME" --retention-in-days 14

echo "== reserved concurrency $RESERVED (global cap on parallel runs)"
aws lambda put-function-concurrency --region "$REGION" --function-name "$NAME" \
  --reserved-concurrent-executions "$RESERVED" >/dev/null 2>&1 || echo "   (account concurrency too low to reserve; relying on in-app global cap)"

echo "== Function URL (auth: $URL_AUTH)"
if aws lambda get-function-url-config --region "$REGION" --function-name "$NAME" >/dev/null 2>&1; then
  aws lambda update-function-url-config --region "$REGION" --function-name "$NAME" --auth-type "$URL_AUTH" >/dev/null
else
  aws lambda create-function-url-config --region "$REGION" --function-name "$NAME" --auth-type "$URL_AUTH" >/dev/null
fi
if [ "$URL_AUTH" = "NONE" ]; then
  aws lambda add-permission --region "$REGION" --function-name "$NAME" --statement-id public-url \
    --action lambda:InvokeFunctionUrl --principal "*" --function-url-auth-type NONE >/dev/null 2>&1 || true
  aws lambda add-permission --region "$REGION" --function-name "$NAME" --statement-id public-url-invoke \
    --action lambda:InvokeFunction --principal "*" --invoked-via-function-url >/dev/null 2>&1 || true
else
  aws lambda remove-permission --region "$REGION" --function-name "$NAME" --statement-id public-url >/dev/null 2>&1 || true
  aws lambda remove-permission --region "$REGION" --function-name "$NAME" --statement-id public-url-invoke >/dev/null 2>&1 || true
fi

if [ -n "${BUDGET_EMAIL:-}" ]; then
  echo "== AWS Budgets alarm (${BUDGET_USD:-10} USD/month, 80% actual + 100% forecast)"
  aws budgets describe-budget --account-id "$ACCOUNT" --budget-name "$NAME-monthly" >/dev/null 2>&1 ||
  aws budgets create-budget --account-id "$ACCOUNT" \
    --budget "{\"BudgetName\":\"$NAME-monthly\",\"BudgetLimit\":{\"Amount\":\"${BUDGET_USD:-10}\",\"Unit\":\"USD\"},\"TimeUnit\":\"MONTHLY\",\"BudgetType\":\"COST\"}" \
    --notifications-with-subscribers "[{\"Notification\":{\"NotificationType\":\"ACTUAL\",\"ComparisonOperator\":\"GREATER_THAN\",\"Threshold\":80,\"ThresholdType\":\"PERCENTAGE\"},\"Subscribers\":[{\"SubscriptionType\":\"EMAIL\",\"Address\":\"$BUDGET_EMAIL\"}]},{\"Notification\":{\"NotificationType\":\"FORECASTED\",\"ComparisonOperator\":\"GREATER_THAN\",\"Threshold\":100,\"ThresholdType\":\"PERCENTAGE\"},\"Subscribers\":[{\"SubscriptionType\":\"EMAIL\",\"Address\":\"$BUDGET_EMAIL\"}]}]"
fi

URL=$(aws lambda get-function-url-config --region "$REGION" --function-name "$NAME" --query FunctionUrl --output text)
echo "== done: $URL  (image $TAG, auth $URL_AUTH)"
