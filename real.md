---
title: リアルタイムコンテスト
subtitle: zLogで参加する
pdf: real.pdf
layout: default
---

{% for file in site.static_files %}
{% if file.basename contains 'real.' and file.extname == '.svg' %}
<img src='{{file.path}}' class='img-thumbnail img-fluid' width='100%'>
{% endif %}
{% endfor %}
