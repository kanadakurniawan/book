import { existsSync, readdirSync, readFileSync, writeFileSync, mkdirSync, copyFileSync, statSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const manuscriptsDir = join(root, 'manuscripts');
const blogDir = dirname(root) + '\\site\\src\\content\\posts';
mkdirSync(blogDir, { recursive: true });

function copyRecursive(src, dest) {
	if (!existsSync(src)) return;
	const entries = readdirSync(src, { withFileTypes: true });
	mkdirSync(dest, { recursive: true });
	for (const entry of entries) {
		const s = join(src, entry.name);
		const d = join(dest, entry.name);
		if (entry.isDirectory()) copyRecursive(s, d);
		else copyFileSync(s, d);
	}
}

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

	const figuresSrc = join(manuscriptsDir, ch, 'figures');
	const figuresDest = join(blogDir, ch, 'figures');
	if (existsSync(figuresSrc)) {
		copyRecursive(figuresSrc, figuresDest);
		console.log(`  SYNC figures/${ch}/ -> site/src/content/posts/${ch}/figures/`);
	}

	const notebooksRoot = join(root, 'notebooks');
	if (existsSync(notebooksRoot)) {
		const prefix = (ch.match(/^ch-\d+/) || [ch])[0] + '-';
		const files = readdirSync(notebooksRoot).filter((f) => f.startsWith(prefix));
		if (files.length > 0) {
			const notebooksDest = join(blogDir, ch, 'notebooks');
			mkdirSync(notebooksDest, { recursive: true });
			for (const f of files) copyFileSync(join(notebooksRoot, f), join(notebooksDest, f));
			console.log(`  SYNC notebooks/${ch}/ -> site/src/content/posts/${ch}/notebooks/`);
		}
	}
}

console.log('Selesai.');