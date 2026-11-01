const list = document.getElementById('issues');
if (list) {
	fetch('/issues.json').then(r => r.json()).then(issues => {
		list.innerHTML = '';
		for (const issue of issues.slice(0, 6)) {
			const item = document.createElement('li');
			const link = document.createElement('a');
			const title = document.createElement('span');
			const meta = document.createElement('span');
			link.href = issue.html_url;
			title.className = 'issue-title';
			title.textContent = issue.title;
			meta.className = 'issue-meta';
			for (const label of issue.labels) {
				const tag = document.createElement('span');
				tag.className = 'label';
				tag.textContent = label.name;
				meta.append(tag);
			}
			meta.append(`#${issue.number} · ${issue.created_at.slice(0, 10)} · ${issue.user.login}`);
			link.append(title, meta);
			item.append(link);
			list.append(item);
		}
	}).catch(() => list.closest('section').querySelector('.issues-fallback').hidden = false);
}
