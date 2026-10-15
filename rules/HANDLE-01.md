id: HANDLE-01
title: Product Handle format
trigger: Every product
kind: deterministic
action: Handle = "<Product title>"; all lower case with blank space replaced with -, and special characters are removed. Replace any run of non-letters/digits with one hyphen
example: "acme-flea-tick-spray-for-dogs-16-oz"
reason: Consistent handle help search and collection pages.
