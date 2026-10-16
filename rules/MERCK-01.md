id: MERCK-01
title: Merck products
kind: deterministic
trigger: Every product; reads the "sku" (supplier item number)
action: custom.is_merck = true if the SKU starts with "012", otherwise false.
example: SKU "012345" -> true; SKU "VPS6" -> false
reason: Merck items are identified by their SKU prefix.
