id: PROP-01
title: Proposition 65
kind: deterministic
trigger: Incoming CSV has two columns. One that is Confirmed Prop 65 and another that is Prop65 itself.  
action: if confirmed = true, the value of compliance.prop_65= prop65 value, otherwise compliance.prop_65 = true. If value is true, Metafield: shopify.disclosure [list.disclosure_reference]=  ["gid://shopify/Metaobject/238268514493"]
reason: Compliance fields required in CA.
note: Illustrative rule for this project, not legal guidance.
