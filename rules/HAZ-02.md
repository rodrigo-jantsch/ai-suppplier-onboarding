id: HAZ-02
title: Aerosol shipping restriction
trigger: Product form is aerosol, or the product page or SDS lists UN1950
action: custom.hazmat = true; add tag "ship-ground-only"; status = needs review
reason: Aerosols can't ship by air.
note: Illustrative rule for this project, not legal guidance.
