---
title: 無線部開発班
subtitle: Journal of Hamradio Informatics
layout: issues
books:
- fava
- mine
- dusk
- chpl
- ats4
---

{% for book in page.books %}
	<div class='row g-2'>
		<a href='/{{book}}' class='col-md-3 col-sm-6'>
			<img src='images/{{book}}.png' class='img-thumbnail' alt='{{book}}'>
		</a>
	</div>
{% endfor %}
