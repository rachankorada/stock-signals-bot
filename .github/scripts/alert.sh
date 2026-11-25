#!/usr/bin/env bash
# Raise an alert as a GitHub issue assigned to the repo owner (GitHub emails them). One open issue per
# title: a repeat adds a comment instead of a new issue. Usage: alert.sh "title" "body"
set -u
title="$1"; body="$2"
repo="$GITHUB_REPOSITORY"; owner="${repo%%/*}"
run_url="$GITHUB_SERVER_URL/$repo/actions/runs/$GITHUB_RUN_ID"
body="$body

Run: $run_url"
gh label create bot-alert -R "$repo" --color B60205 --description "Problems the trading bot raised" --force >/dev/null 2>&1
existing=$(gh issue list -R "$repo" --label bot-alert --state open --search "\"$title\" in:title" --json number,title \
           --jq ".[] | select(.title == \"$title\") | .number" | head -n1)
if [ -n "$existing" ]; then
  gh issue comment "$existing" -R "$repo" --body "$body"
else
  gh issue create -R "$repo" --title "$title" --label bot-alert --assignee "$owner" --body "$body"
fi
