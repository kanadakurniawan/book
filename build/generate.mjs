import { execSync } from 'node:child_process';
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const manuscriptsDir = join(root, 'manuscripts');
const releasesDir = join(root, 'releases');

function hasTool(name) {
	try {
		execSync(`${name} --version`, { stdio: 'ignore' });
		return true;
	} catch {
		return false;
	}
}

const args = process.argv.slice(2);
const versionArg = args.find((a) => a.startsWith('--version='));
if (!versionArg) {
	console.error('Usage: node build/generate.mjs --version=v1.0.0');
	process.exit(1);
}
const version = versionArg.split('=')[1];
const releaseDir = join(releasesDir, version);
mkdirSync(releaseDir, { recursive: true });

const hasPandoc = hasTool('pandoc');
const hasLatex = hasTool('pdflatex') || hasTool('xelatex') || hasTool('lualatex');

console.log(`Generating release ${version}`);
console.log(`  pandoc: ${hasPandoc ? 'OK' : 'TIDAK ADA'} | LaTeX: ${hasLatex ? 'OK' : 'TIDAK ADA'}`);

const chapterDirs = readdirSync(manuscriptsDir, { withFileTypes: true })
	.filter((d) => d.isDirectory())
	.map((d) => d.name);

if (chapterDirs.length === 0) {
	console.warn('Tidak ada bab di manuscripts/ — tambahkan folder ch-0N-<slug>/master.md');
	process.exit(0);
}

for (const ch of chapterDirs) {
	const master = join(manuscriptsDir, ch, 'master.md');
	const bib = join(manuscriptsDir, ch, 'refs.bib');
	if (!existsSync(master)) {
		console.warn(`  SKIP ${ch}: master.md tidak ditemukan`);
		continue;
	}
	const outBase = join(releaseDir, ch);
	if (hasPandoc) {
		const bibOpts = existsSync(bib) ? `--citeproc --bibliography="${bib}"` : '';
		try {
			execSync(`pandoc "${master}" ${bibOpts} -o "${outBase}.docx"`, { stdio: 'ignore' });
			console.log(`  OK  ${ch}.docx`);
		} catch (e) {
			console.error(`  GAGAL docx ${ch}: ${e.message}`);
		}
		if (hasLatex) {
			try {
				execSync(`pandoc "${master}" ${bibOpts} -o "${outBase}.pdf" --pdf-engine=xelatex`, {
					stdio: 'ignore',
				});
				console.log(`  OK  ${ch}.pdf`);
			} catch (e) {
				console.error(`  GAGAL pdf ${ch}: ${e.message}`);
			}
		} else {
			console.warn(`  SKIP ${ch}.pdf (LaTeX tidak terpasang)`);
		}
	} else {
		console.warn(`  SKIP ${ch}: pandoc tidak terpasang. Install dari https://pandoc.org`);
	}
	writeFileSync(join(releaseDir, 'MANIFEST.md'), `# Release ${version}\n\nDirilis: ${new Date().toISOString()}\n`);
}

console.log(`Selesai. Rilis: ${releaseDir}`);