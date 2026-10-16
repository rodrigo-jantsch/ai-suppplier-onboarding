id: DRO-01
title: Dropship
kind: deterministic
trigger: Supplier attribute "Is Dropship" = Y
action: variant custom.dropship = true; add tag "GMC: Dropship"
reason: Dropship items have a different delivery estimate.
note: Illustrative rule for this project, not legal guidance.
