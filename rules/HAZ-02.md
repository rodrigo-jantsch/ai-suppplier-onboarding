id: HAZ-02
title: Hazardous shipping restriction
kind: deterministic
trigger: Supplier attribute "Jeffers Is Hazardous" = Y (also check "Is DOT/ORMD" = Y for variant custom.ormd)
action: variant custom.hazardous = true; if "Is DOT/ORMD" = Y then variant custom.ormd = true
reason: Hazardous items have shipping restrictions.
note: Illustrative rule for this project, not legal guidance.
