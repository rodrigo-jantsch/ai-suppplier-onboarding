id: BIO-01
title: Biologicals (vaccines)
kind: deterministic
trigger: Supplier attribute "Jeffers Is Biological" = Y
action: variant custom.biological = true; add tag "GMC: Vaccine Shipping"; status = needs review
reason: Biologicals ship with ice packs and coolers.
note: Illustrative rule for this project, not legal guidance.
