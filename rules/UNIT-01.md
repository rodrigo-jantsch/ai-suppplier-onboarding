id: UNIT-01
title: Size and unit normalization
kind: deterministic
trigger: Every product with a size or weight
action: Sizes go to custom.sizes. Liquids in "fl oz" -> "<n> oz", milliliters -> "<n> ml", liters -> "<n> L"; dry goods in "lb" (convert kg x 2.2046, round to 1 decimal); counts as "<n> count". The variant weight (variant.weight_lb) is always in pounds.
examples: "16OZ" -> "16 oz"; "13.6KG" -> "30 lb"; "60CT" -> "60 count"; "1 liter" -> "1 L"
reason: Shoppers filter and compare by size.
