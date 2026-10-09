// Teams for checking the engine's team validator against Pokémon Showdown's.
//
//   node oracle/gen_teams.js --n 2000 --seed 1 --out teams.jsonl [--format gen9championsvgc2026regmc]
//       Random teams, most of them legal or one step from legal, each with Showdown's verdict.
//   node oracle/gen_teams.js --judge sampled.jsonl [--format ...]
//       Showdown's verdict on teams the engine made (`teamcheck --sample`): prints those it refuses.
//
// The teams are built from Showdown's data directly (which species exist, what they
// learn), not from the regulation file, so that the file is not checked against itself.
'use strict';
const fs = require('fs');
const L = require('./lib.js');
const { PS } = L;

const args = {};
for (let i = 2; i < process.argv.length; i++) {
	const a = process.argv[i];
	if (a.startsWith('--')) args[a.slice(2)] = process.argv[i + 1] && !process.argv[i + 1].startsWith('--') ? process.argv[++i] : true;
}
const format = PS.Dex.formats.get(args.format || L.FORMAT);
if (!format.exists) throw new Error('no such format: ' + args.format);
const dex = PS.Dex.forFormat(format);
const validator = PS.TeamValidator.get(format);
const judge = team => validator.validateTeam(JSON.parse(JSON.stringify(team))) || [];
/**
 * Showdown's verdict, and for a team it accepts, the team as its validator leaves it. The
 * validator repairs what it can rather than refuse it: a Mega written as the species
 * becomes the ordinary forme holding its stone, an impossible gender is corrected. The
 * engine's validator does not repair, so it is shown what Showdown accepted in the end.
 */
function judgeAndRepair(team) {
	const repaired = JSON.parse(JSON.stringify(team));
	const problems = validator.validateTeam(repaired) || [];
	if (problems.length) return { problems, shown: team };
	if (judge(repaired).length) throw new Error('a team Showdown accepted and repaired is refused afterwards:\n' + JSON.stringify(team));
	return { problems, shown: repaired };
}

if (args.judge) {
	let n = 0;
	let refused = 0;
	for (const line of fs.readFileSync(args.judge, 'utf8').split('\n')) {
		if (!line.trim()) continue;
		n++;
		const team = JSON.parse(line);
		const problems = judge(team);
		if (problems.length) {
			refused++;
			if (refused <= 5) console.log(`team ${n}: ${problems.join(' / ')}\n    ${line.slice(0, 400)}`);
		}
	}
	console.log(`${n} teams made by the engine; Showdown refuses ${refused}`);
	process.exit(refused ? 1 : 0);
}

const N = parseInt(args.n || '1000');
let seed = parseInt(args.seed || '1') >>> 0;
const rand = () => {
	// mulberry32
	seed = (seed + 0x6D2B79F5) >>> 0;
	let t = seed;
	t = Math.imul(t ^ (t >>> 15), t | 1);
	t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
	return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};
const int = n => Math.floor(rand() * n);
const pick = list => list[int(list.length)];
const shuffled = list => {
	const a = list.slice();
	for (let i = a.length - 1; i > 0; i--) {
		const j = int(i + 1);
		[a[i], a[j]] = [a[j], a[i]];
	}
	return a;
};

/** What a species learns in this regulation's data (a forme without an entry of its own uses its base species'). */
function learnable(species) {
	let ls = dex.species.getLearnsetData(species.id);
	if (!(ls && ls.learnset)) ls = dex.species.getLearnsetData(dex.species.get(species.baseSpecies).id);
	if (!(ls && ls.learnset)) return [];
	return Object.keys(ls.learnset).filter(id => dex.moves.get(id).exists && !dex.moves.get(id).isNonstandard);
}

const everySpecies = dex.species.all();
const everyMove = dex.moves.all().filter(m => m.id !== 'struggle');
const everyAbility = dex.abilities.all();
const everyItem = dex.items.all();
// What looks bringable from the data alone; the validator has the last word.
const plausible = everySpecies.filter(s => !s.isNonstandard && !s.isMega && !s.battleOnly &&
	!s.tags.includes('Mythical') && !s.tags.includes('Restricted Legendary') && learnable(s).length);
const plainItems = everyItem.filter(i => !i.isNonstandard);
const natures = dex.natures.all().map(n => n.name).filter(n => n !== 'Serious');

function spread() {
	const sp = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
	let left = 66;
	const keys = Object.keys(sp);
	while (left > 0) {
		const k = pick(keys);
		const add = Math.min(left, 32 - sp[k], 1 + int(16));
		sp[k] += add;
		left -= add;
	}
	return sp;
}

function plausibleSet(species, usedItems) {
	const moves = shuffled(learnable(species)).slice(0, 1 + int(4)).map(id => dex.moves.get(id).name);
	const abilities = Object.values(species.abilities);
	let item = '';
	if (rand() < 0.85) {
		const free = plainItems.filter(i => !usedItems.has(i.id));
		const stones = free.filter(i => i.megaStone && i.megaStone[species.name]);
		item = (stones.length && rand() < 0.4 ? pick(stones) : pick(free)).name;
		usedItems.add(PS.toID(item));
	}
	return {
		name: species.name, species: species.name, item, ability: pick(abilities), moves,
		nature: pick(natures), evs: spread(), level: 50,
		gender: species.gender || (rand() < 0.3 ? '' : pick(['M', 'F'].filter(g => species.genderRatio[g] > 0))),
	};
}

function plausibleTeam(size) {
	const team = [];
	const nums = new Set();
	const usedItems = new Set();
	while (team.length < size) {
		const s = pick(plausible);
		if (nums.has(s.num)) continue;
		nums.add(s.num);
		team.push(plausibleSet(s, usedItems));
	}
	return team;
}

// One way of breaking a team each. Some do not break it (a random move may be
// one the Pokémon learns); the validator says which.
const FAULTS = {
	otherSpecies(team) { const s = pick(team); const to = pick(everySpecies); s.species = to.name; s.name = to.name; },
	otherForme(team) {
		const s = pick(team);
		const formes = everySpecies.filter(o => o.num === dex.species.get(s.species).num);
		const to = pick(formes); s.species = to.name; s.name = to.name;
	},
	megaAsSpecies(team) {
		const megas = everySpecies.filter(o => o.isMega && !o.isNonstandard);
		const to = pick(megas); const s = pick(team);
		s.species = to.name; s.name = to.name;
		if (rand() < 0.5) s.item = to.requiredItem || s.item;
	},
	sameSpeciesTwice(team) {
		const [a, b] = shuffled(team);
		if (!b) return;
		const formes = everySpecies.filter(o => o.num === dex.species.get(a.species).num && !o.isMega && !o.battleOnly);
		const to = pick(formes);
		Object.assign(b, plausibleSet(to, new Set(team.map(t => PS.toID(t.item)))));
	},
	otherMove(team) { const s = pick(team); s.moves[int(Math.max(1, s.moves.length))] = pick(everyMove).name; },
	moveTwice(team) { const s = pick(team); if (s.moves.length) s.moves.push(pick(s.moves)); },
	fiveMoves(team) {
		const s = pick(team);
		const learn = learnable(dex.species.get(s.species)).map(id => dex.moves.get(id).name).filter(m => !s.moves.includes(m));
		while (s.moves.length < 5 && learn.length) s.moves.push(learn.pop());
	},
	noMoves(team) { pick(team).moves = []; },
	otherAbility(team) { pick(team).ability = pick(everyAbility).name; },
	noAbility(team) { pick(team).ability = ''; },
	otherItem(team) { pick(team).item = pick(everyItem).name; },
	itemTwice(team) { const [a, b] = shuffled(team.filter(s => s.item)); if (a && b) b.item = a.item; },
	noItems(team) { for (const s of team) s.item = ''; },
	tooManyInOneStat(team) { const s = pick(team); s.evs = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 }; s.evs[pick(Object.keys(s.evs))] = 33 + int(30); },
	tooManyInAll(team) { const s = pick(team); s.evs = { hp: 12, atk: 12, def: 12, spa: 12, spd: 12, spe: 7 + int(6) }; },
	exactlyTheLimit(team) { const s = pick(team); s.evs = { hp: 32, atk: 32, def: 2, spa: 0, spd: 0, spe: 0 }; },
	negativePoints(team) { const s = pick(team); s.evs.hp = -1 - int(5); },
	fewPoints(team) { const s = pick(team); s.evs = { hp: 1 + int(4), atk: 0, def: 0, spa: 0, spd: 0, spe: 0 }; },
};
const faultNames = Object.keys(FAULTS);

const out = fs.createWriteStream(args.out || 'teams.jsonl');
const tally = {};
let legal = 0;
for (let n = 0; n < N; n++) {
	// Mostly full teams; now and then the wrong number of Pokémon.
	const sizeRoll = rand();
	const size = sizeRoll < 0.9 ? 6 : pick([1, 2, 3, 4, 5, 7, 8]);
	const team = plausibleTeam(size);
	const faults = [];
	const roll = rand();
	const count = roll < 0.35 ? 0 : roll < 0.9 ? 1 : 2;
	for (let k = 0; k < count; k++) {
		const f = pick(faultNames);
		FAULTS[f](team);
		faults.push(f);
	}
	if (size !== 6) faults.push('size' + size);
	const { problems, shown } = judgeAndRepair(team);
	if (!problems.length) legal++;
	for (const f of faults.length ? faults : ['none']) {
		tally[f] = tally[f] || [0, 0];
		tally[f][problems.length ? 1 : 0]++;
	}
	out.write(JSON.stringify({ id: n, faults, legal: !problems.length, problems, team: shown }) + '\n');
}
out.end();
console.log(`${format.name}: ${N} teams, ${legal} legal`);
console.log('  fault: legal/refused  ' + Object.entries(tally).map(([f, [a, b]]) => `${f} ${a}/${b}`).join(', '));
