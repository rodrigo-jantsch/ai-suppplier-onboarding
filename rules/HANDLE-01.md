id: HANDLE-01
title: Product handle format
kind: deterministic
trigger: Every product
action: Handle is generated from the Title: lowercase, then replace any run of characters that are not letters or digits with a single hyphen, and trim hyphens from both ends.
example: "TiaGard® 12.5% (tiamulin hydrogen fumarate) Liquid Concentrate for Swine Dysentery & Pneumonia Treatment" -> "tiagard-12-5-tiamulin-hydrogen-fumarate-liquid-concentrate-for-swine-dysentery-pneumonia-treatment"
reason: Consistent handles help search and collection pages.
