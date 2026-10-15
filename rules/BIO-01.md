id: BIO-01
title: Vaccines
kind: deterministic
trigger: Product has a requires refrigeration or biologican identification in the incoming CSV.
action: custom.biological=true; add tag "GMC: Vaccine Shipping";, status = needs review;
reason: Vaccine has to be shipped with Ice pack and coolers.
note: Illustrative rule for this project, not legal guidance.
