id: PROP-01
title: Proposition 65
kind: deterministic
trigger: Every product; reads supplier attributes "Jeffers Is Prop 65" (value) and "Jeffers Prop 65 Validated" (confirmed)
action: If "Jeffers Prop 65 Validated" is Y, compliance.prop_65 = value of "Jeffers Is Prop 65" (1/Y -> true, 0/N -> false). Otherwise compliance.prop_65 = true. When compliance.prop_65 is true, set shopify.disclosure = ["gid://shopify/Metaobject/238268514493"].
reason: Compliance fields required in CA; unconfirmed items default to showing the warning.
note: Illustrative rule for this project, not legal guidance.
