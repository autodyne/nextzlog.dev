name = ARGV[0]

# zenn.dev
HOST = 'https://zenn.dev/nextzlog/books'

# source files
path = File.dirname(__FILE__)
root = File.dirname(path)
book = File.join(path, 'book.cls')
file = File.join(path, "#{name}.tex")
post = File.join(path, "#{name}.pdf")

# modules
require 'yaml'
require 'fileutils'
require File.join(root, 'nomd.jar')
java_import 'engine.NoMD'

# call TeXt to convert LaTeX into Markdown
text = NoMD.process([book, file].to_java(:String))
text = text.gsub('```sample', '```')
text = text.gsub('```chapel', '```')
text = text.gsub('```dlang', '```d')
text = text.gsub(/\\{(\$)?/, '\\lbrace\1 ')
text = text.gsub(/\\}(\$)?/, '\\rbrace\1 ')
text = text.gsub('/scales/', '/images/')
text = text.gsub(/\\text\{(.*)\$(.*)\$/, '\\text{\1\\(\2\\)')
text = text.gsub(/\t/, '  ')
subs = text.split(/^## /)[1..]
conf = text.split(/^---/)[1].strip
yaml = YAML.load(conf)
path = File.join(name, yaml['subtitle'].downcase.gsub(/\W+/, '-'))
desc = YAML.load(File.read(File.join(root, 'pages/_config.yml')))['briefs'][name]

# output abstract
FileUtils.mkdir_p(name)
File.open(File.join(name, sprintf('%s.md', name)), mode='w') do |file|
	file.puts('---')
	file.puts(conf)
	file.puts("pdf: #{name}.pdf")
	file.puts("web: #{HOST}/#{File.basename(path)}")
	file.puts('---')
	file.puts('{% for file in site.static_files %}')
	file.puts("{% if file.basename contains '#{name}.' and file.extname == '.svg' %}")
	file.puts('<img src="{{file.path}}" class="img-thumbnail img-fluid" width="100%">')
	file.puts('{% endif %}')
	file.puts('{% endfor %}')
end

exit if subs.empty?

# output config
FileUtils.mkdir_p(path)
File.open(File.join(path, 'config.yaml'), mode='w') do |file|
	YAML.dump(yaml.merge({
		'topics' => yaml['topics'].split(','),
		'summary' => desc,
		'chapters' => (1..subs.size).map(&:to_s),
		'published' => true,
	}), file)
end

# output chapters
subs.each.with_index(1) do |body, index|
	File.open(File.join(path, sprintf('%d.md', index)), mode='w') do |file|
		file.puts('---')
		file.puts("title: #{body.lines.first.strip}")
		file.puts('---')
		file.puts("## #{body}")
	end
end

# create cover
system("pdf2svg #{post} cover.svg 1")
system("inkscape --export-png=#{File.join(path, 'cover.png')} cover.svg")
