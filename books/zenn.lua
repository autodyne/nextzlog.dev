-- pandoc lua books/zenn.lua NAME OUTPUT
-- converts books/NAME.tex into a Zenn book under OUTPUT/books with its cover from books/NAME.pdf

local name, output = arg[1], arg[2] or 'zenn'
local root = pandoc.path.directory(arg[0])

local function slurp(path)
	local file = assert(io.open(path))
	local text = file:read('a')
	file:close()
	return text
end

local function spit(path, text)
	pandoc.system.make_directory(pandoc.path.directory(path), true)
	local file = assert(io.open(path, 'w'))
	file:write(text)
	file:close()
end

-- balanced group starting at position i, opened by open and closed by close

local function group(text, i, open, close)
	if text:sub(i, i) ~= open then
		return nil
	end
	local depth = 0
	for j = i, #text do
		local c = text:sub(j, j)
		if c == '\\' then
			goto continue
		elseif c == open and text:sub(j - 1, j - 1) ~= '\\' then
			depth = depth + 1
		elseif c == close and text:sub(j - 1, j - 1) ~= '\\' then
			depth = depth - 1
			if depth == 0 then
				return text:sub(i + 1, j - 1), j + 1
			end
		end
		::continue::
	end
end

local function blank(text, i)
	return text:match('^%s*()', i)
end

-- macros defined by \NewDocumentCommand and \RenewDocumentCommand

local macros = {}

local skip = {
	book = true, header = true, chapter = true, maketitle = true,
	sum = true, prod = true, frac = true, contentsname = true, appendixname = true,
}

local function define(text)
	for start in text:gmatch('\\%a+DocumentCommand()') do
		local id, i = text:match('^\\?(@?%a+)()', blank(text, start))
		local spec, j = group(text, blank(text, i), '{', '}')
		local body = spec and select(1, group(text, blank(text, j), '{', '}'))
		if body and not id:find('@') and not skip[id] then
			local args = {}
			local k = 1
			while k <= #spec do
				local kind = spec:sub(k, k)
				local default, l = group(spec, k + 1, '{', '}')
				if kind:match('[moO]') then
					table.insert(args, {kind = kind, default = default or ''})
				end
				k = kind == 'O' and l or k + 1
			end
			macros[id] = {args = args, body = body}
		end
	end
end

local function operators(text)
	for id, body in text:gmatch('\\DeclareMathOperator%*?{\\(%a+)}(%b{})') do
		macros[id] = {args = {}, body = '\\mathop{\\mathrm' .. body .. '}'}
	end
end

local function aliases(text)
	for id, target in text:gmatch('\\LetLtxMacro\\(%a+)\\(%a+)') do
		if not macros[target] then
			macros[id] = {args = {}, body = '\\' .. target}
		end
	end
end

local function call(text, i, macro)
	local values = {}
	for _, arg in ipairs(macro.args) do
		local value, j
		if arg.kind == 'm' then
			i = blank(text, i)
			value, j = group(text, i, '{', '}')
			if not value then
				value, j = text:match('^(\\%a+)()', i)
			end
			if not value then
				value, j = text:sub(i, i), i + 1
			end
		else
			value, j = group(text, i, '[', ']')
			value = value or (arg.kind == 'O' and arg.default or '-NoValue-')
		end
		table.insert(values, value)
		i = j or i
	end
	local body = macro.body:gsub('#(%d)', function(n)
		return values[tonumber(n)]
	end)
	return body, i
end

local function ifvalue(text)
	text = text:gsub('\\mathchoice(%b{})%b{}%b{}%b{}', function(display)
		return display:sub(2, -2)
	end)
	return (text:gsub('\\IfValueTF(%b{})(%b{})(%b{})', function(value, yes, no)
		return (value == '{-NoValue-}' and no or yes):sub(2, -2)
	end))
end

local function expand(text)
	for _ = 1, 8 do
		local parts, last, changed = {}, 1, false
		local i = 1
		while true do
			local s, id, e = text:match('()\\(%a+)()', i)
			if not s then
				break
			end
			local macro = macros[id]
			if macro then
				local body, j = call(text, e, macro)
				table.insert(parts, text:sub(last, s - 1))
				table.insert(parts, body)
				last, i, changed = j, j, true
			else
				i = e
			end
		end
		table.insert(parts, text:sub(last))
		text = ifvalue(table.concat(parts))
		if not changed then
			break
		end
	end
	return text
end

-- verbatim regions are kept away from macro expansion

local stash = {}

local function protect(text)
	text = text:gsub('\\begin{Verbatim}(%b{})(.-)\\end{Verbatim}', function(lang, code)
		table.insert(stash, {lang = lang:sub(2, -2), title = '', code = code:gsub('^\n', '')})
		return string.format('\\begin{verbatim}STASH%d\\end{verbatim}', #stash)
	end)
	return text
end

-- algorithmic environments become pseudo code

local function algorithmic(body)
	local lines, depth = {}, 0
	local function put(line)
		table.insert(lines, string.rep('  ', depth) .. line)
	end
	for line in body:gmatch('[^\n]+') do
		line = line:gsub('%$', ''):gsub('\\mathrm(%b{})', function(t) return t:sub(2, -2) end)
		local head, rest = line:match('^%s*\\(%a+)%s*(.*)$')
		if head == 'Procedure' then
			local f, a = rest:match('^(%b{})(%b{})')
			put(string.format('procedure %s(%s)', f:sub(2, -2), a:sub(2, -2)))
			depth = depth + 1
		elseif head == 'For' or head == 'If' then
			put(string.format(head == 'For' and 'for %s do' or 'if %s then', rest:match('^(%b{})'):sub(2, -2)))
			depth = depth + 1
		elseif head == 'Else' then
			depth = depth - 1
			put('else')
			depth = depth + 1
		elseif head and head:match('^End') then
			depth = depth - 1
			put('end')
		elseif head == 'State' then
			put(rest)
		elseif head == 'Return' then
			put('return ' .. rest)
		end
	end
	table.insert(stash, {lang = '', title = '', code = table.concat(lines, '\n') .. '\n', algorithm = true})
	return string.format('\\begin{verbatim}STASH%d\\end{verbatim}', #stash)
end

-- subfloats are numbered within each figure and table

local sublabels = {}

local function subfloats(env, body)
	local count = 0
	body = body:gsub('\\subfloat%[(.-)%](%b{})', function(caption, content)
		count = count + 1
		local mark = '(' .. count .. ')'
		caption = caption:gsub('\\label{(.-)}', function(id)
			sublabels[id] = mark
			return ''
		end)
		content = content:sub(2, -2)
		if content:find('\\begin{tabular}') then
			return string.format('\n\n%s %s\n\n%s\n\n', mark, caption, content)
		end
		return string.format('\n\n%s\n\n%s %s\n\n', content, mark, caption)
	end)
	if env == 'table' and count > 0 then
		return '\\begin{figure}\\tablefloat' .. body .. '\\end{figure}'
	end
	return '\\begin{' .. env .. '}' .. body .. '\\end{' .. env .. '}'
end

local function prepare(text)
	text = protect(text)
	text = text:gsub('\\begin{algorithmic}(.-)\\end{algorithmic}', algorithmic)
	text = text:gsub('\\begin{algorithm}%b[]', '\\begin{figure}')
	text = text:gsub('\\begin{algorithm}', '\\begin{figure}'):gsub('\\end{algorithm}', '\\end{figure}')
	text = text:gsub('\\begin{minipage}%b{}', ''):gsub('\\end{minipage}', '')
	text = text:gsub('\\begin{center}', ''):gsub('\\end{center}', '')
	text = text:gsub('\\begin{tabular}(%b{})', function(spec)
		spec = spec:sub(2, -2):gsub('[<>@]%b{}', ''):gsub('p%b{}', 'l'):gsub('[DBT]', 'l'):gsub('C', 'c'):gsub('[^lcr]', '')
		return '\\begin{tabular}{' .. spec .. '}'
	end)
	text = text:gsub('\\midrule%b[]', '\\midrule'):gsub('\\textregistered%s*', '®')
	text = text:gsub('\\infer%[.-%]', '\\infer')
	local count
	repeat
		text, count = text:gsub('\\infer(%b{})(%b{})', function(conclusion, premises)
			return string.format('\\frac{\\begin{matrix}%s\\end{matrix}}{%s}', premises:sub(2, -2), conclusion:sub(2, -2))
		end)
	until count == 0
	text = expand(text)
	text = text:gsub('\\texttt(%b{})', function(code)
		return '\\texttt' .. code:gsub('%-%-', '-{}-')
	end)
	text = text:gsub('\\oldeqref', '\\ref'):gsub('\\subref', '\\ref'):gsub('\\ALG@name{}', 'Algorithm')
	for _, env in ipairs({'figure', 'table'}) do
		text = text:gsub('\\begin{' .. env .. '}(.-)\\end{' .. env .. '}', function(body)
			return subfloats(env, body)
		end)
	end
	return text
end

-- numbering of chapters, sections, equations, figures, tables and algorithms

local languages = {sample = '', chapel = '', dlang = 'd', ['c++'] = 'cpp'}

local function code(item)
	local lang = item.lang:lower()
	lang = languages[lang] or lang
	if item.title ~= '' then
		lang = lang .. ':' .. item.title
	end
	return pandoc.RawBlock('markdown', '```' .. lang .. '\n' .. item.code:gsub('\t', '  ') .. '```')
end

local function number(doc)
	local labels, chapter, counts = sublabels, 0, {}
	local function next(kind)
		counts[kind] = (counts[kind] or 0) + 1
		return string.format('%d.%d', chapter, counts[kind])
	end
	local function label(text)
		return text:match('\\label{(.-)}')
	end
	local function unlabel(text)
		return (text:gsub('%s*\\label{.-}', ''))
	end
	local function take(element, n)
		if element.identifier and element.identifier ~= '' then
			labels[element.identifier] = n
		end
		return element:walk({
			Span = function(span)
				if span.attributes.label then
					labels[span.identifier] = n
					return span.content
				end
			end,
			RawInline = function(raw)
				local id = label(raw.text)
				if id and raw.text:match('^\\label') then
					labels[id] = n
					return {}
				end
			end,
		})
	end

	-- first pass assigns numbers to labels

	local first = {
		Header = function(h)
			if h.level == 1 then
				chapter, counts = chapter + 1, {}
				h.attributes.number = tostring(chapter)
			elseif h.level == 2 and not h.classes:includes('unnumbered') then
				h.attributes.number = next('section')
			end
			return take(h, h.attributes.number or '?')
		end,
		Math = function(m)
			if m.mathtype == 'DisplayMath' then
				local n = next('equation')
				local id = label(m.text)
				if id then
					labels[id] = n
				end
				m.text = unlabel(m.text):gsub('%s+$', '') .. ' \\qquad(' .. n .. ')'
				return m
			end
		end,
		Figure = function(f)
			local kind = 'figure'
			f.content:walk({
				RawInline = function(r)
					if r.text:find('\\tablefloat', 1, true) then
						kind = 'table'
					end
				end,
				RawBlock = function(r)
					if r.text:find('\\tablefloat', 1, true) then
						kind = 'table'
					end
				end,
				CodeBlock = function(c)
					local n = c.text:match('^STASH(%d+)$')
					if n and stash[tonumber(n)].algorithm then
						kind = 'algorithm'
					end
				end,
			})
			local n = next(kind)
			f = take(f, n)
			f.attributes.number = n
			f.attributes.kind = kind
			return f
		end,
		Table = function(t)
			if #t.caption.long == 0 then
				return t
			end
			local n = next('table')
			t = take(t, n)
			t.attributes.number = n
			return t
		end,
	}
	doc = doc:walk({traverse = 'topdown', Header = first.Header, Math = first.Math, Figure = first.Figure, Table = first.Table})

	-- second pass resolves references and renders numbered elements

	local function reference(link)
		if link.attributes['reference-type'] then
			local n = labels[link.attributes.reference] or '?'
			return pandoc.Str(n)
		end
	end
	return doc:walk({
		Link = reference,
		RawInline = function(raw)
			local id = raw.text:match('^\\ref{(.-)}$')
			if id then
				return pandoc.Str(labels[id] or '?')
			end
		end,
		Header = function(h)
			local n = h.attributes.number
			local content = n and {pandoc.Str(n), pandoc.Space(), table.unpack(h.content)} or h.content
			return pandoc.Header(h.level + 1, content)
		end,
		Math = function(m)
			local text = m.text:gsub('\\{', '\\lbrace '):gsub('\\}', '\\rbrace '):gsub('\\bm([^%a])', '\\boldsymbol%1')
			text = text:gsub('^%s+', ''):gsub('%s+$', '')
			text = text:gsub('\\idotsint', '\\int\\cdots\\int'):gsub('\\hfill?', '')
			text = text:gsub('\\text(%b{})', function(t)
				return '\\text' .. t:gsub('%$(.-)%$', '\\(%1\\)')
			end)
			if m.mathtype == 'DisplayMath' then
				return pandoc.RawInline('markdown', '\n\n$$' .. text .. '$$\n\n')
			end
			return pandoc.RawInline('markdown', ' $' .. text .. '$ ')
		end,
		Image = function(img)
			img.src = '/images/' .. pandoc.path.filename(img.src):gsub('%.eps$', '.png'):gsub('%.pdf$', '.png')
			return pandoc.Image({pandoc.Str(img.src:sub(2))}, img.src)
		end,
		Figure = function(f)
			local blocks = pandoc.Blocks({})
			local prefix = ({algorithm = 'Algorithm ', table = 'Table ', figure = 'Fig. '})[f.attributes.kind]
			for _, block in ipairs(f.content) do
				blocks:insert(block)
			end
			local caption = pandoc.Para({pandoc.Str(prefix .. f.attributes.number), pandoc.Space(), table.unpack(f.caption.long[1] and f.caption.long[1].content or {})})
			blocks:insert(f.attributes.kind == 'table' and 1 or #blocks + 1, caption)
			return blocks
		end,
		Table = function(t)
			local n, text = t.attributes.number, pandoc.utils.stringify(t.caption.long)
			t.caption = {}
			t.attr = pandoc.Attr()
			t = t:walk({
				Para = function(p) return pandoc.Plain(p.content) end,
				RawBlock = function(r) if r.format == 'latex' then return {} end end,
				RawInline = function(r) if r.format == 'latex' then return {} end end,
			})
			local rows = pandoc.List({})
			for _, body in ipairs(t.bodies) do
				rows:extend(body.body)
			end
			t.bodies = {{attr = pandoc.Attr(), body = rows, head = {}, row_head_columns = 0}}
			for i, spec in ipairs(t.colspecs) do
				t.colspecs[i] = {spec[1], pandoc.ColWidthDefault}
			end
			if not n then
				return t
			end
			return {pandoc.Para({pandoc.Str('Table ' .. n), pandoc.Space(), pandoc.Str(text)}), t}
		end,
		CodeBlock = function(c)
			local n = c.text:match('^STASH(%d+)$')
			if n then
				return code(stash[tonumber(n)])
			end
		end,
	})
end

-- main

local tex = slurp(pandoc.path.join({root, name .. '.tex'}))
define(slurp(pandoc.path.join({root, 'book.cls'})))
define(tex)
operators(slurp(pandoc.path.join({root, 'book.cls'})))
operators(tex)
aliases(slurp(pandoc.path.join({root, 'book.cls'})))

local title, subtitle, keywords = tex:match('\\book(%b{})(%b{})%b{}(%b{})')
title, subtitle = pandoc.utils.stringify(pandoc.read(title, 'latex')), pandoc.utils.stringify(pandoc.read(subtitle, 'latex'))
keywords = keywords:sub(2, -2)

local body = tex:match('\\begin{document}(.-)\\end{document}')
body = body:gsub('\\maketitle', ''):gsub('\\tableofcontents', '')
local doc = pandoc.read(prepare(body), 'latex+raw_tex-smart')
doc = number(doc)

local chapters, current = {}, nil
for _, block in ipairs(doc.blocks) do
	if block.t == 'Header' and block.level == 2 then
		current = {title = pandoc.utils.stringify(block.content), blocks = pandoc.Blocks({})}
		table.insert(chapters, current)
	end
	if current then
		current.blocks:insert(block)
	end
end

local slug = subtitle:lower():gsub('[^%w]+', '-')
local dir = pandoc.path.join({output, 'books', slug})
local options = {extensions = {'pipe_tables', 'raw_html'}, wrap_text = 'preserve'}
for i, chapter in ipairs(chapters) do
	local text = pandoc.write(pandoc.Pandoc(chapter.blocks), 'gfm', options)
	text = text:gsub('\n\n\n+', '\n\n')
	spit(pandoc.path.join({dir, i .. '.md'}), string.format('---\ntitle: "%s"\n---\n\n%s', chapter.title, text))
end

local config = {
	'title: "' .. title .. '"',
	'summary: "https://nextzlog.dev/' .. name .. '.pdf"',
	'topics:',
}
for topic in keywords:gmatch('[^,]+') do
	table.insert(config, '  - "' .. topic .. '"')
end
table.insert(config, 'published: true')
table.insert(config, 'chapters:')
for i = 1, #chapters do
	table.insert(config, '  - "' .. i .. '"')
end
spit(pandoc.path.join({dir, 'config.yaml'}), table.concat(config, '\n') .. '\n')

local pdf = pandoc.path.join({root, name .. '.pdf'})
pandoc.pipe('pdftoppm', {'-png', '-singlefile', '-scale-to', '500', pdf, pandoc.path.join({dir, 'cover'})}, '')
