id: TAG-01
title: Product Tags
trigger: Every product
kind: deterministic
action: Add the parent item number from the input spreadsheet (column "Variation Parent SKU") as the tag "CatalogRef: <Parent Item Number>" (no hyphen, one space after the colon). Always add the tag "New Arrival" as well.
example: Parent item number VPS5 -> tags ["CatalogRef: VPS5", "New Arrival"]
reason: Useful to identify products.
