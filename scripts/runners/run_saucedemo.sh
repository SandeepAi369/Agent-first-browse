#!/bin/bash
cd "$(dirname "$0")/../.." || exit 1  # repo root
./agent.sh "Go to https://www.saucedemo.com. Log in specifically with the username 'problem_user' and password 'secret_sauce'. Your task is to find the 'Sauce Labs Fleece Jacket', add ONLY this jacket to the cart, go to the cart page, and definitively verify that the Fleece Jacket is actually in the cart."
