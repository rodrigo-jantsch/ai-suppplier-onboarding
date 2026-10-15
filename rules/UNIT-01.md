id: UNIT-01
title: Size and unit normalization
kind: deterministic
trigger: Every product with a size or weight
action: Liquids in "fl oz" or "ml" -> store as "<n> oz" or "<n> ml"; dry goods in "lb" (convert kg x 2.2046, round to 1 decimal); counts as "<n> count"; custom.weight_lb is always pounds, liters → "<n> L"
examples: "16OZ" -> "16 oz"; "13.6KG" -> "30 lb"; "60CT" -> "60 count"; 2 liters → "2 L"
reason: Shoppers filter and compare by size.
