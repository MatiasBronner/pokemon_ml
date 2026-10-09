// Plays random battles in Pokémon Showdown and records, for every decision,
// the choices made, what was legal, and the full state afterwards. The Rust
// engine replays the file from the same seed and must match at every step.
//
//   node gen_cases.js --n 200 --seed 1 --out cases.jsonl [--stats stats.json] [--trace] [--only ID]
//                     [--max-turns 250] [--policy switch] [--plain]
//
// --plain gives every Pokémon no ability, no item and no gender (the set-up the
// engine's first version was checked with).
'use strict';
const fs = require('fs');
const path = require('path');
const L = require('./lib.js');
const { PS, dex } = L;

const args = {};
for (let i = 2; i < process.argv.length; i++) {
	const a = process.argv[i];
	if (!a.startsWith('--')) continue;
	const next = process.argv[i + 1];
	if (next === undefined || next.startsWith('--')) args[a.slice(2)] = true;
	else args[a.slice(2)] = process.argv[++i];
}
const N = parseInt(args.n || '100');
const SEED = parseInt(args.seed || '1');
const OUT = args.out || path.join(__dirname, 'cases.jsonl');
const TRACE = !!args.trace;
const MAX_TURNS = parseInt(args['max-turns'] || '250');
const ONLY = args.only !== undefined ? parseInt(args.only) : null;
const POLICY = args.policy || 'mixed';
const PLAIN = !!args.plain;

const pool = JSON.parse(fs.readFileSync(path.join(__dirname, 'pool.json'), 'utf8'));
const ALL_MOVES = [...new Set(pool.species.flatMap(s => s.moves))].sort();
const STATS = args.stats || null;
const tally = { moves: {}, events: {}, abilities: {}, items: {}, brought: { abilities: {}, items: {} } };
const bump = (table, key) => { table[key] = (table[key] || 0) + 1; };
function tallyLog(log) {
	for (const line of log) {
		const parts = line.split('|');
		const kind = parts[1];
		// Anything in the log that names an ability or item counts as it doing something visible.
		for (const m of line.matchAll(/ability: ([^|\]]+)/g)) bump(tally.abilities, PS.toID(m[1]));
		for (const m of line.matchAll(/item: ([^|\]]+)/g)) bump(tally.items, PS.toID(m[1]));
		if (kind === '-ability') bump(tally.abilities, PS.toID(parts[3]));
		if (kind === '-enditem' || kind === '-item') bump(tally.items, PS.toID(parts[3]));
		if (kind === 'move') bump(tally.moves, PS.toID(parts[3]));
		else if (kind === '-status' || kind === 'cant') bump(tally.events, `${kind} ${parts[3]}`);
		else if (kind === '-activate' && parts[3]) bump(tally.events, `-activate ${parts[3]}`);
		else if (['-crit', '-miss', '-fail', '-immune', 'switch', 'faint', '-hitcount', '-heal', '-boost', '-unboost',
			'-supereffective', '-resisted', '-curestatus', '-singleturn', 'win', 'tie'].includes(kind)) bump(tally.events, kind);
	}
}
const CHOOSABLE = new Set(['normal', 'any', 'adjacentAlly', 'adjacentAllyOrSelf', 'adjacentFoe']);

function pick(rand, arr) { return arr[Math.floor(rand() * arr.length)]; }
function shuffled(rand, arr) {
	const a = arr.slice();
	for (let i = a.length - 1; i > 0; i--) {
		const j = Math.floor(rand() * (i + 1));
		[a[i], a[j]] = [a[j], a[i]];
	}
	return a;
}

/** A random legal Champions stat-point spread: 66 points, at most 32 per stat. */
function randomSpread(rand) {
	const sp = [0, 0, 0, 0, 0, 0];
	let left = 66;
	const style = rand();
	if (style < 0.5) {
		// The common shape: two stats maxed, the remainder in a third.
		const order = shuffled(rand, [0, 1, 2, 3, 4, 5]);
		sp[order[0]] = 32; sp[order[1]] = 32; sp[order[2]] = 2;
		return sp;
	}
	while (left > 0) {
		const k = Math.floor(rand() * 6);
		const add = Math.min(left, 32 - sp[k], 1 + Math.floor(rand() * 12));
		sp[k] += add; left -= add;
	}
	return sp;
}

/**
 * Ability, item and gender for a set. Showdown's simulator does not check that a
 * species can legally have an ability, so half the time any modelled ability is
 * used: that exercises abilities on bodies and movesets their real owners lack.
 */
function extras(rand, s) {
	if (PLAIN) return { ability: 'noability', item: '', gender: 'N' };
	let ability = 'noability';
	const r = rand();
	if (pool.abilities.length && r >= 0.06) {
		ability = (r < 0.5 && s.abilities.length) ? pick(rand, s.abilities) : pick(rand, pool.abilities);
	}
	const item = (pool.items.length && rand() < 0.8) ? pick(rand, pool.items) : '';
	const gender = s.gender || (rand() < 0.5 ? 'M' : 'F');
	return { ability, item, gender };
}

function randomSet(rand) {
	const s = pick(rand, pool.species);
	const moves = shuffled(rand, s.moves).slice(0, 4);
	// Protect is on nearly every real doubles set; make sure it is exercised.
	if (rand() < 0.35 && !moves.includes('protect') && s.moves.includes('protect')) moves[3] = 'protect';
	return { species: s.id, moves, nature: pick(rand, pool.natures), sp: randomSpread(rand), ...extras(rand, s) };
}

/** A set guaranteed to know `moveId`, so that every modelled move gets exercised. */
function featuredSet(rand, moveId) {
	const learners = pool.species.filter(s => s.moves.includes(moveId));
	if (!learners.length) return randomSet(rand);
	const s = pick(rand, learners);
	const moves = [moveId, ...shuffled(rand, s.moves.filter(m => m !== moveId)).slice(0, 3)];
	return { species: s.id, moves, nature: pick(rand, pool.natures), sp: randomSpread(rand), ...extras(rand, s) };
}

function buildTeams(rand) {
	const mode = rand();
	const team = () => {
		const out = [];
		const seen = new Set();
		while (out.length < 6) {
			const set = randomSet(rand);
			if (seen.has(set.species)) continue;
			seen.add(set.species);
			out.push(set);
		}
		return out;
	};
	if (mode < 0.03) {
		// Nobody can attack: PP runs dry and the battle is settled by Struggle.
		const passive = pool.species.map(s => ({ id: s.id, moves: s.moves.filter(m => dex.moves.get(m).category === 'Status') }))
			.filter(s => s.moves.length >= 4);
		const set = () => {
			const s = pick(rand, passive);
			const full = pool.species.find(x => x.id === s.id);
			return { species: s.id, moves: shuffled(rand, s.moves).slice(0, 4), nature: pick(rand, pool.natures), sp: randomSpread(rand), ...extras(rand, full) };
		};
		return [Array.from({ length: 6 }, set), Array.from({ length: 6 }, set)];
	}
	if (mode < 0.15) {
		// Everyone shares one species and spread: every ordering decision is a speed tie.
		const base = randomSet(rand);
		const sp = pool.species.find(s => s.id === base.species);
		const clone = () => ({ ...base, moves: shuffled(rand, sp.moves).slice(0, 4) });
		return [Array.from({ length: 6 }, clone), Array.from({ length: 6 }, clone)];
	}
	if (mode < 0.35) {
		// Mirror match: each Pokémon ties with its counterpart.
		const t = team();
		return [t, t.map(s => ({ ...s }))];
	}
	return [team(), team()];
}

function toPsSet(set, i) {
	const species = dex.species.get(set.species);
	const evs = {};
	L.STAT_IDS.forEach((k, j) => { evs[k] = set.sp[j]; });
	return {
		name: `${species.baseSpecies}${i}`, species: species.name,
		item: set.item ? dex.items.get(set.item).name : '', ability: dex.abilities.get(set.ability).name,
		moves: set.moves.map(m => dex.moves.get(m).name), nature: set.nature,
		// Gender must be fixed: an unset gender is rolled from the battle's RNG.
		gender: set.gender, evs, ivs: { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 }, level: 50,
	};
}

function snapshot(battle) {
	const winner = !battle.ended ? null : battle.winner === 'P1' ? 0 : battle.winner === 'P2' ? 1 : -1;
	return {
		turn: battle.turn,
		ended: battle.ended,
		winner,
		request: battle.requestState || '',
		rng: battle.prng.getSeed().split(',').map(Number),
		effect_order: battle.effectOrder,
		sides: battle.sides.map(side => ({
			left: side.pokemonLeft,
			mons: side.pokemon.map(p => ({
				idx: p.pickIndex,
				species: p.species.id,
				hp: p.hp,
				maxhp: p.maxhp,
				stats: [p.maxhp, ...L.STAT_IDS.slice(1).map(k => p.storedStats[k])],
				status: p.fainted ? '' : p.status,
				time: p.fainted ? 0 : (p.statusState.time || 0),
				stage: p.fainted ? 0 : (p.statusState.stage || 0),
				boosts: L.BOOST_IDS.map(k => p.boosts[k]),
				pp: p.moveSlots.map(m => m.pp),
				active: p.isActive,
				fainted: p.fainted,
				switch_flag: !!p.switchFlag,
				speed: p.speed,
				vol: Object.keys(p.volatiles).map(id => `${id}:${p.volatiles[id].duration || 0}:${volDetail(id, p.volatiles[id])}`),
				ability: p.ability,
				item: p.item,
				last_item: p.lastItem,
				used_item: !!p.usedItemThisTurn,
				ate_berry: !!p.ateBerry,
				types: p.types,
				trapped: p.trapped ? (p.trapped === 'hidden' ? 2 : 1) : 0,
				disabled: p.moveSlots.map(m => !!m.disabled),
				ability_order: p.abilityState.effectOrder || 0,
				item_order: p.itemState.effectOrder || 0,
				active_turns: p.activeTurns,
				move_result: [p.moveThisTurnResult, p.moveLastTurnResult].map(resultCode).join(''),
			})),
		})),
	};
}

/** The state a volatile carries, as the Rust engine prints it. */
function volDetail(id, state) {
	switch (id) {
	case 'stall': return state.counter || 0;
	case 'choicelock': return state.move || 0;
	case 'confusion': return state.time || 0;
	case 'metronome': return `${state.lastMove || '-'}/${state.numConsecutive || 0}`;
	default: return 0;
	}
}
function resultCode(v) {
	return v === undefined ? 'u' : v === null ? 'n' : v === true ? 't' : v === false ? 'f' : '?';
}

/**
 * Per active slot, every choice Showdown would accept, as canonical choice strings.
 * This reads the Pokémon's real state rather than the request sent to the player,
 * because the request deliberately hides some things (a trap not yet revealed).
 */
function legalOptions(battle, side) {
	const req = side.activeRequest;
	if (!req || req.wait) return [['pass'], ['pass']];
	const bench = [];
	for (let i = side.active.length; i < side.pokemon.length; i++) if (!side.pokemon[i].fainted) bench.push(i);
	if (req.forceSwitch) {
		const need = req.forceSwitch.filter(Boolean).length;
		return req.forceSwitch.map(f => {
			if (!f) return ['pass'];
			const opts = bench.map(i => `switch ${i + 1}`);
			if (bench.length < need) opts.push('pass');
			return opts;
		});
	}
	return req.active.map((a, pos) => {
		const p = side.active[pos];
		if (p.fainted) return ['pass'];
		const opts = [];
		const moves = p.getMoves();
		// No usable move left: any move choice becomes Struggle.
		if (!moves.length) opts.push('move 1');
		moves.forEach((m, j) => {
			if (m.disabled) return;
			if (CHOOSABLE.has(m.target)) {
				for (const loc of [1, 2, -1, -2]) if (battle.validTargetLoc(loc, p, m.target)) opts.push(`move ${j + 1} ${loc}`);
			} else {
				opts.push(`move ${j + 1}`);
			}
		});
		if (!p.trapped) for (const i of bench) opts.push(`switch ${i + 1}`);
		return opts;
	});
}

function chooseFor(rand, options) {
	if (options.some(opts => opts.includes('pass') && opts.length > 1)) {
		// Fewer replacements than empty slots: hand the bench out at random, pass the rest.
		const open = shuffled(rand, options.map((opts, i) => i).filter(i => options[i].length > 1));
		const bench = shuffled(rand, options[open[0]].filter(o => o !== 'pass'));
		const picks = options.map(() => 'pass');
		open.forEach((slot, k) => { if (k < bench.length) picks[slot] = bench[k]; });
		return picks;
	}
	// Favour attacking so battles finish, but switch often enough to exercise it.
	// `--policy switch` instead switches whenever possible, which never ends a battle
	// and so runs into Showdown's 1000-turn limit.
	const attackRate = POLICY === 'switch' ? 0 : 0.88;
	for (let attempt = 0; attempt < 50; attempt++) {
		const picks = options.map(opts => {
			const moves = opts.filter(o => o.startsWith('move'));
			const others = opts.filter(o => !o.startsWith('move'));
			if (moves.length && (!others.length || rand() < attackRate)) return pick(rand, moves);
			return pick(rand, others);
		});
		const switches = picks.filter(p => p.startsWith('switch'));
		if (new Set(switches).size !== switches.length) continue;
		return picks;
	}
	throw new Error('could not find a consistent choice');
}

function stackLabel() {
	const lines = new Error().stack.split('\n').slice(3, 9)
		.map(l => (l.match(/at (?:new )?([\w.$<>]+)/) || [])[1])
		.filter(x => x && !/^(PRNG|Gen5RNG)\./.test(x) && x !== 'Battle.random' && x !== 'Battle.randomChance' && x !== 'Battle.sample');
	return lines.slice(0, 4).join(' < ');
}

function runCase(id) {
	const rand = L.mulberry32((SEED * 1000003 + id) | 0);
	const [full1, full2] = buildTeams(rand);
	const seed = [0, 0, 0, 0].map(() => Math.floor(rand() * 65536));
	const picks = [full1, full2].map(full => shuffled(rand, full.map((_, i) => i)).slice(0, 4));
	// Each battle features one modelled move on a lead, cycling through all of them.
	const featured = ALL_MOVES[id % ALL_MOVES.length];
	const fside = Math.floor(id / ALL_MOVES.length) % 2;
	[full1, full2][fside][picks[fside][0]] = featuredSet(rand, featured);
	// Likewise one modelled ability and one modelled item, on leads, so each gets its share of battles.
	if (!PLAIN) {
		if (pool.abilities.length) {
			const set = [full1, full2][1 - fside][picks[1 - fside][0]];
			[full1, full2][1 - fside][picks[1 - fside][0]] = { ...set, ability: pool.abilities[id % pool.abilities.length] };
		}
		if (pool.items.length) {
			const set = [full1, full2][fside][picks[fside][1]];
			[full1, full2][fside][picks[fside][1]] = { ...set, item: pool.items[id % pool.items.length] };
		}
	}

	const battle = new PS.Battle({ formatid: L.FORMAT, seed: seed.join(',') });
	let draws = [];
	if (TRACE) {
		const orig = battle.prng.random.bind(battle.prng);
		battle.prng.random = (from, to) => {
			const v = orig(from, to);
			draws.push(`rng(${to ? `${from},${to}` : from})=${v}  ${stackLabel()}`);
			return v;
		};
	}
	battle.setPlayer('p1', { name: 'P1', team: full1.map(toPsSet) });
	battle.setPlayer('p2', { name: 'P2', team: full2.map(toPsSet) });
	if (battle.requestState !== 'teampreview') throw new Error(`expected team preview, got ${battle.requestState}`);

	// Remember each Pokémon's pick order; Showdown reorders side.pokemon as it switches.
	battle.sides.forEach((side, s) => picks[s].forEach((orig, k) => { side.pokemon[orig].pickIndex = k; }));
	battle.sides.forEach((side, s) => {
		if (!battle.choose(side.id, `team ${picks[s].map(i => i + 1).join(',')}`)) throw new Error(side.choice.error);
	});
	const teams = [full1, full2].map((full, s) => picks[s].map(i => full[i]));
	for (const t of teams) for (const set of t) { bump(tally.brought.abilities, set.ability); if (set.item) bump(tally.brought.items, set.item); }

	let logPos = battle.log.length;
	const out = { id, seed, teams, initial: snapshot(battle), steps: [], truncated: false };
	if (TRACE) { out.initial.draws = draws; out.initial.log = battle.log.slice(0); draws = []; }
	while (!battle.ended) {
		if (battle.turn > MAX_TURNS) { out.truncated = true; break; }
		const legal = battle.sides.map(side => legalOptions(battle, side));
		const choices = legal.map(opts => chooseFor(rand, opts));
		// Decide who is being asked before anyone answers: the last answer starts the next
		// request, which would otherwise look like one this step still had to fill.
		const asked = battle.sides.map(side => !!side.activeRequest && !side.activeRequest.wait);
		battle.sides.forEach((side, s) => {
			if (!asked[s]) return;
			if (!battle.choose(side.id, choices[s].join(', '))) {
				throw new Error(`case ${id}: Showdown rejected "${choices[s].join(', ')}" for ${side.id}: ${side.choice.error}`);
			}
		});
		const after = snapshot(battle);
		if (TRACE) { after.draws = draws; draws = []; after.log = battle.log.slice(logPos); }
		logPos = battle.log.length;
		out.steps.push({ choices: choices.map(c => c.join(', ')), legal, after });
	}
	if (STATS) tallyLog(battle.log);
	return out;
}

const fd = fs.openSync(OUT, 'w');
let steps = 0, truncated = 0, turns = 0;
const ids = ONLY !== null ? [ONLY] : Array.from({ length: N }, (_, i) => i);
for (const id of ids) {
	const c = runCase(id);
	steps += c.steps.length;
	if (c.truncated) truncated++;
	turns += c.steps.length ? c.steps[c.steps.length - 1].after.turn : 0;
	fs.writeSync(fd, JSON.stringify(c) + '\n');
}
fs.closeSync(fd);
if (STATS) {
	const unused = ALL_MOVES.filter(m => !tally.moves[m]);
	const least = ALL_MOVES.map(m => [m, tally.moves[m] || 0]).sort((a, b) => a[1] - b[1]).slice(0, 10);
	fs.writeFileSync(STATS, JSON.stringify({
		battles: ids.length, decisions: steps, modelled_moves: ALL_MOVES.length, unused_moves: unused, least_used: least,
		events: tally.events, moves: tally.moves,
		// For each modelled ability and item: [times brought, log lines naming it].
		abilities: Object.fromEntries(pool.abilities.map(a => [a, [tally.brought.abilities[a] || 0, tally.abilities[a] || 0]])),
		items: Object.fromEntries(pool.items.map(a => [a, [tally.brought.items[a] || 0, tally.items[a] || 0]])),
	}, null, 1));
	console.log(`moves used at least once: ${ALL_MOVES.length - unused.length}/${ALL_MOVES.length}; least used: ${least.slice(0, 5).map(x => x.join(' x')).join(', ')}`);
}
console.log(`wrote ${ids.length} battles, ${steps} decisions, ${truncated} cut off at turn ${MAX_TURNS}, mean length ${(turns / ids.length).toFixed(1)} turns -> ${OUT}`);
