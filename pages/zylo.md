---
title: ZyLO
subtitle: Golangで広げるコンテストロガーの可能性
pdf: zylo.pdf
---

{% for file in site.static_files %}
{% if file.basename contains 'zylo.' and file.extname == '.svg' %}
<img src='{{file.path}}' class='img-thumbnail img-fluid' width='100%'>
{% endif %}
{% endfor %}
