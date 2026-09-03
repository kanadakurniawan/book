import { watch, existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { syncAll } from './sync-to-blog.mjs';

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const watchDirs = [
	join(root, 'manuscripts'),
	join(root, 'notebooks'),
];

let timer = null;
function scheduleSync(reason) {
	clearTimeout(timer);
	timer = setTimeout(() => {
		console.log(`\n[watch] Perubahan terdeteksi (${reason}). Menjalankan sync...`);
		const n = syncAll();
		console.log(`[watch] Selesai. ${n} bab disinkronkan.`);
	}, 400);
}

for (const dir of watchDirs) {
	if (!existsSync(dir)) {
		console.warn(`[watch] Folder tidak ditemukan: ${dir}`);
		continue;
	}
	watch(dir, { recursive: true }, (_event, filename) => {
		if (!filename) return;
		const name = String(filename);
		if (name.endsWith('.md') || name.endsWith('.ipynb') || name.includes('figures')) {
			scheduleSync(name);
		}
	});
	console.log(`[watch] Memantau: ${dir}`);
}

console.log('[watch] Berjalan. Tekan Ctrl+C untuk berhenti.');
console.log('[watch] Setiap perubahan pada manuscripts/ atau notebooks/ akan otomatis disinkronkan ke blog (site/src/content/posts/).\n');
