// Shared helpers for the Showdown oracle scripts.
'use strict';
const path = require('path');
const PS = require(path.join(__dirname, 'pokemon-showdown', 'dist', 'sim'));

const MOD = 'champions';
const FORMAT = 'gen9championsvgc2026regmc';
const dex = PS.Dex.mod(MOD);

const STAT_IDS = ['hp', 'atk', 'def', 'spa', 'spd', 'spe'];
const BOOST_IDS = ['atk', 'def', 'spa', 'spd', 'spe', 'accuracy', 'evasion'];
const STATUSES = ['brn', 'par', 'psn', 'tox', 'slp', 'frz'];

// Move properties every move object carries; they are either read directly by
// the generator or irrelevant to battle mechanics.
const BASE_KEYS = new Set([
	'name', 'id', 'fullname', 'effectType', 'exists', 'num', 'gen', 'isNonstandard', 'duration', 'noCopy',
	'affectsFainted', 'status', 'weather', 'sourceEffect', 'type', 'target', 'basePower', 'accuracy', 'critRatio',
	'baseMoveType', 'secondary', 'secondaries', 'hasSheerForceBoost', 'priority', 'category',
	'overrideOffensiveStat', 'overrideOffensivePokemon', 'overrideDefensiveStat', 'overrideDefensivePokemon',
	'ignoreNegativeOffensive', 'ignorePositiveDefensive', 'ignoreOffensive', 'ignoreDefensive', 'ignoreImmunity',
	'pp', 'noPPBoosts', 'isZ', 'isMax', 'flags', 'selfSwitch', 'ignoreAbility', 'damage', 'spreadHit', 'forceSTAB',
	'volatileStatus', 'maxMove', 'zMove', 'contestType', 'desc', 'shortDesc', 'realMove', 'nonGhostTarget',
	'pressureTarget',
]);
// Extra declarative properties the Rust engine implements.
const EXTRA_OK = new Set(['boosts', 'self', 'drain', 'recoil', 'heal', 'thawsTarget', 'willCrit', 'ignoreEvasion', 'multihit']);
const TARGETS_OK = new Set([
	'normal', 'any', 'adjacentFoe', 'allAdjacentFoes', 'allAdjacent', 'self', 'adjacentAlly', 'adjacentAllyOrSelf', 'allies',
]);
const BAD_FLAGS = ['charge', 'recharge', 'futuremove', 'cantusetwice', 'mustpressure', 'pledgecombo'];
// Moves with script callbacks that the Rust engine implements by hand.
const SPECIAL = { protect: 'Protect', detect: 'Protect', struggle: 'Struggle' };

function hitEffectReasons(e, where, why) {
	for (const k in e) {
		if (k === 'chance' || k === 'dustproof') continue;
		if (k === 'boosts') continue;
		if (k === 'status') {
			if (!STATUSES.includes(e.status)) why.push(`${where}.status=${e.status}`);
		} else if (k === 'volatileStatus') {
			if (e.volatileStatus !== 'flinch') why.push(`${where}.volatile=${e.volatileStatus}`);
		} else if (k === 'self') {
			for (const sk in e.self) if (sk !== 'boosts') why.push(`${where}.self.${sk}`);
		} else {
			why.push(`${where}.${k}`);
		}
	}
}

/** Returns [] if the Rust engine models every effect of this move, else the reasons it does not. */
function unsupportedReasons(m) {
	if (SPECIAL[m.id]) return [];
	const why = [];
	for (const k in m) {
		if (BASE_KEYS.has(k) || EXTRA_OK.has(k)) continue;
		why.push((typeof m[k] === 'function' ? 'fn:' : 'key:') + k);
	}
	if (!TARGETS_OK.has(m.target)) why.push('target:' + m.target);
	if (m.status && !STATUSES.includes(m.status)) why.push('status:' + m.status);
	if (m.volatileStatus) why.push('volatile:' + m.volatileStatus);
	if (m.weather) why.push('weather');
	if (m.selfSwitch) why.push('selfSwitch');
	if (m.damage) why.push('damage:' + m.damage);
	if (m.isZ || m.isMax) why.push('zmax');
	if (m.forceSTAB) why.push('forceSTAB');
	if (m.ignoreImmunity && m.ignoreImmunity !== true) why.push('ignoreImmunity:map');
	if (m.ignoreImmunity === true && m.category !== 'Status') why.push('ignoreImmunity');
	if (m.ignoreNegativeOffensive || m.ignorePositiveDefensive || m.ignoreOffensive) why.push('ignoreBoosts');
	if (m.overrideDefensivePokemon) why.push('overrideDefensivePokemon');
	for (const f of BAD_FLAGS) if (m.flags[f]) why.push('flag:' + f);
	if (m.self) for (const sk in m.self) if (sk !== 'boosts' && sk !== 'chance') why.push('self.' + sk);
	if (m.secondaries) for (const s of m.secondaries) hitEffectReasons(s, 'sec', why);
	return why;
}

/** Species that can be brought to a Champions battle (Megas are reached in battle, not brought). */
function legalSpecies() {
	return dex.species.all().filter(s => s.exists && !s.isNonstandard && s.tier !== 'Illegal');
}

function learnableMoves(species) {
	// Champions learnsets are stored per species; formes without their own entry use the base species'.
	let ls = dex.species.getLearnsetData(species.id);
	if (!(ls && ls.learnset)) ls = dex.species.getLearnsetData(dex.species.get(species.baseSpecies).id);
	if (!(ls && ls.learnset)) return [];
	return Object.keys(ls.learnset).filter(id => {
		const m = dex.moves.get(id);
		return m.exists && !m.isNonstandard;
	}).sort();
}

/** Every move the generated Rust table contains, in table order. */
function tableMoves() {
	const ids = new Set(dex.moves.all().filter(m => m.exists && !m.isNonstandard).map(m => m.id));
	for (const s of legalSpecies()) for (const m of learnableMoves(s)) ids.add(m);
	ids.add('struggle');
	return [...ids].sort().map(id => dex.moves.get(id));
}

/** Every species the generated Rust table contains, in table order. */
function tableSpecies() {
	return dex.species.all().filter(s => s.exists && !s.isNonstandard).sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
}

/** Small deterministic generator for the harness itself (never the battle's RNG). */
function mulberry32(a) {
	return function () {
		a |= 0; a = (a + 0x6D2B79F5) | 0;
		let t = Math.imul(a ^ (a >>> 15), 1 | a);
		t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

module.exports = {
	PS, dex, MOD, FORMAT, STAT_IDS, BOOST_IDS, STATUSES, SPECIAL,
	unsupportedReasons, legalSpecies, learnableMoves, tableMoves, tableSpecies, mulberry32,
};
