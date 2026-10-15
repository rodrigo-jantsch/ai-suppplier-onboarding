id: TAX-01
title: Product category (Shopify taxonomy) selection
kind: guidance
trigger: Every product
action: Choose exactly one category from config/taxonomy.yaml; never invent or edit a category name. Pick the most specific category the product data clearly supports. If no subcategory fits, choose its parent level (entries marked fallback: true). Never choose a deeper category than the evidence supports. If the product is for animals but is not a pet product (livestock, equine, poultry), use "Animals & Pet Supplies". If the confidence is low, choose the parent level and set confidence to low so the item goes to review.
examples: Flea and tick pump spray for dogs -> "... > Pet Flea & Tick Control > Flea & Tick Sprays"; pet medicine with no matching subcategory -> "... > Pet Supplies > Pet Medicine"; swine water-soluble antibiotic -> "Animals & Pet Supplies"
reason: The category must be one Shopify offers. A correct general category is a small fix for the reviewer; a wrong specific one breaks collections, filters and marketplace feeds.
