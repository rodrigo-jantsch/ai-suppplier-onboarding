id: DRO-01
title: Dropship
kind: deterministic
trigger: Productis not shipped by us. Marked as dropship in the incoming CSV.
action: custom.dropship=true; add tag "GMC: Dropship";
reason: Dropship has different ETA
note: Illustrative rule for this project, not legal guidance.
