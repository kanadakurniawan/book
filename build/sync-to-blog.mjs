import { existsSync, readdirSync, readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const manuscriptsDir = join(root, 'manuscripts');
const blogDir = dirname(root) + '\\site\\src\\content\\posts';
mkdirSync(blogDir, { recursive: true });

function parseFrontmatter(text) {
	const m = text.match(/^---\r?\n([\s\S]*?)\r?\n---/);
	if (!m) return null;
	const out = {};
	for (const line of m[1].split(/\r?\n/)) {
		const i = line.indexOf(':');
		if (i < 0) continue;
		const key = line.slice(0, i).trim();
		const value = line.slice(i + 1).trim();
		out[key] = value;
	}
	return { frontmatter: out, rest: text.slice(m[0].length) };
}

const chapterDirs = readdirSync(manuscriptsDir, { withFileTypes: true })
	.filter((d) => d.isDirectory())
	.map((d) => d.name);

if (chapterDirs.length === 0) {
	console.log('Tidak ada bab untuk disinkronkan.');
	process.exit(0);
}

function unquote(value) {
	const v = value?.trim() ?? '';
	if (v.length >= 2 && ((v[0] === '"' && v[v.length - 1] === '"') || (v[0] === "'" && v[v.length - 1] === "'"))) {
		return v.slice(1, -1);
	}
	return v;
}

function q(s) {
	return JSON.stringify(s);
}

for (const ch of chapterDirs) {
	const master = join(manuscriptsDir, ch, 'master.md');
	if (!existsSync(master)) continue;
	const raw = readFileSync(master, 'utf8');
	const parsed = parseFrontmatter(raw);
	if (!parsed) {
		console.warn(`SKIP ${ch}: frontmatter YAML tidak ditemukan`);
		continue;
	}
	const fm = parsed.frontmatter;
	const draft = fm.status === 'draft';
	const title = unquote(fm.title);
	const description = unquote(fm.description);
	const categories = (fm.categories || '[]').replace(/^\[/, '').replace(/\]$/, '');
	const tags = (fm.tags || '[]').replace(/^\[/, '').replace(/\]$/, '');
	const synced = [
		'---',
		`title: ${q(title)}`,
		`description: ${q(description)}`,
		`pubDate: ${fm.pubDate || '2026-01-01'}`,
		`categories: [${categories}]`,
		`tags: [${tags}]`,
		`draft: ${draft}`,
		'---',
		parsed.rest,
	].join('\n');

	const target = join(blogDir, `${ch}.md`);
	writeFileSync(target, synced);
	console.log(`  SYNC ${ch}.md -> site/src/content/posts/${ch}.md ${draft ? '(draft)' : ''}`);
}

console.log('Selesai.');