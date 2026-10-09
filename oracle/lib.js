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
			if (!MOVE_VOLATILES.includes(e.volatileStatus)) why.push(`${where}.volatile=${e.volatileStatus}`);
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
	if (m.volatileStatus && !MOVE_VOLATILES.includes(m.volatileStatus)) why.push('volatile:' + m.volatileStatus);
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

// ---- abilities, items and conditions ------------------------------------------

// Volatile conditions the Rust engine implements, in `VolKind` order.
const VOLATILES = ['protect', 'stall', 'flinch', 'confusion', 'choicelock', 'gem', 'metronome', 'flashfire', 'unburden'];
// Volatiles a move may inflict through `volatileStatus` (its own or a secondary's).
const MOVE_VOLATILES = ['flinch', 'confusion'];

// Abilities and items whose every callback has a hand-written body in the Rust
// engine (src/abilities.rs, src/items.rs). Everything else is rejected by
// `Battle::new`. DEFERRED_* records why something is not here yet.
// Abilities: every ability a Champions species can have, except the ones listed
// here with the mechanic they wait for.
const DEFERRED_ABILITIES = {};
const defer = (why, ids) => { for (const id of ids.split(' ')) DEFERRED_ABILITIES[id] = why; };
defer('weather', 'chlorophyll cloudnine drizzle drought dryskin forecast hydration icebody iceface leafguard megasol ' +
	'raindish sandforce sandrush sandspit sandstream sandveil slushrush snowcloak snowwarning solarpower swiftswim');
defer('terrain', 'electricsurge grasspelt grassysurge mimicry psychicsurge seedsower surgesurfer');
defer('forme changes', 'battlebond disguise gulpmissile hungerswitch shieldsdown stancechange zerotohero terashell');
defer('Illusion and Transform', 'illusion imposter');
defer('the Disable volatile', 'cursedbody');
defer('the Attract volatile', 'cutecharm');
defer('the Charge volatile', 'electromorphosis');
defer('entry hazards', 'toxicdebris');
defer('switching out mid-turn', 'emergencyexit wimpout');
const SUPPORTED_ABILITIES = new Set(['noability']);
// Modelled effects with a part that can never come up yet, because it reacts to
// something that is not modelled. They are exact for every battle the engine
// accepts; this list says what to revisit when the missing mechanic arrives.
const DORMANT_PARTS = {
	abilities: {
		anticipation: 'only writes to the battle log',
		aromaveil: 'blocks Taunt, Encore, Disable, Torment, Attract and Heal Block, none of which is modelled',
		damp: 'blocks self-destructing moves, which are not modelled (its Aftermath block is)',
		embodyaspectcornerstone: 'needs Terastallization, which Champions does not have',
		embodyaspecthearthflame: 'needs Terastallization, which Champions does not have',
		embodyaspectteal: 'needs Terastallization, which Champions does not have',
		embodyaspectwellspring: 'needs Terastallization, which Champions does not have',
		flowerveil: 'its Yawn block (Yawn is not modelled)',
		forewarn: 'only writes to the battle log (its random pick is still drawn)',
		frisk: 'only writes to the battle log',
		gluttony: 'only matters for pinch berries, which are not in Champions',
		guarddog: 'its block on being forced out (forced switches are not modelled)',
		harvest: 'always succeeding in sun (weather is not modelled)',
		heavymetal: 'weight is only read by moves that are not modelled',
		infiltrator: 'bypasses Substitute and screens, which are not modelled',
		insomnia: 'its Yawn block (Yawn is not modelled)',
		lightmetal: 'weight is only read by moves that are not modelled',
		magicbounce: 'bouncing moves that target a side (none modelled)',
		oblivious: 'its Attract and Taunt immunity (neither is modelled)',
		overcoat: 'its weather-damage immunity (weather is not modelled)',
		parentalbond: 'its Secret Power special case (not in Champions)',
		purifyingsalt: 'its Yawn block (Yawn is not modelled)',
		sapsipper: 'absorbing Grass moves that target a side (none modelled)',
		screencleaner: 'removes screens, which are not modelled',
		soundproof: 'blocking sound moves that target a side (none modelled)',
		stickyhold: 'its Knock Off block (Knock Off is not modelled)',
		sturdy: 'its one-hit-KO immunity (those moves are not modelled)',
		suctioncups: 'blocks being forced out (forced switches are not modelled)',
		sweetveil: 'its Yawn block (Yawn is not modelled)',
		vitalspirit: 'its Yawn block (Yawn is not modelled)',
	},
	items: {
		bigroot: 'boosting Leech Seed, Ingrain, Aqua Ring and Strength Sap (only draining moves are modelled)',
		bindingband: 'boosts binding moves, which are not modelled',
		damprock: 'extends rain (weather is not modelled)',
		heatrock: 'extends sun (weather is not modelled)',
		icyrock: 'extends snow (weather is not modelled)',
		lightclay: 'extends screens, which are not modelled',
		mentalherb: 'cures Taunt, Encore, Disable, Torment, Attract and Heal Block, none of which is modelled',
		smoothrock: 'extends sandstorm (weather is not modelled)',
		terrainextender: 'extends terrain, which is not modelled',
	},
};
// Items: everything except the ones listed here with the mechanic they wait for. (A Mega Stone is
// accepted on any Pokémon; whether the Mega it leads to is modelled is checked when the battle is built.)
const DEFERRED_ITEMS = {
	ejectbutton: 'switching out mid-turn',
	redcard: 'forced switching',
	electricseed: 'terrain',
	grassyseed: 'terrain',
	mistyseed: 'terrain',
	psychicseed: 'terrain',
};
const SUPPORTED_ITEMS = new Set(['']);
for (const item of dex.items.all()) {
	if (!item.exists || item.isNonstandard) continue;
	if (!DEFERRED_ITEMS[item.id]) SUPPORTED_ITEMS.add(item.id);
}

/** Event names, in the order of the Rust `Ev` enum (src/data.rs is the single source). */
const EVENTS = (() => {
	const src = require('fs').readFileSync(path.join(__dirname, '..', 'src', 'data.rs'), 'utf8');
	const body = src.match(/pub enum Ev \{([\s\S]*?)\n\}/)[1];
	return body.split('\n').map(l => l.replace(/\/\/.*$/, '').trim().replace(/,$/, '')).filter(Boolean);
})();
const PREFIXES = ['Ally', 'Foe', 'Any', 'Source'];
const META = /(Priority|Order|SubOrder)$/;

/**
 * The event callbacks of an ability, item or condition, with the ordering
 * metadata `Battle#resolvePriority` would give them.
 */
function callbacks(effect) {
	const out = [];
	const subOrderDefault = () => {
		if (effect.effectType === 'Condition') return 2;
		if (effect.effectType === 'Ability') {
			if (effect.name === 'Poison Touch' || effect.name === 'Perish Body') return 6;
			if (effect.name === 'Stall') return 9;
			return 7;
		}
		if (effect.effectType === 'Item') return 8;
		return 0;
	};
	const meta = key => ({
		order: effect[key + 'Order'] || 0,
		priority: effect[key + 'Priority'] || 0,
		subOrder: effect[key + 'SubOrder'] || subOrderDefault(),
	});
	const parse = rest => {
		for (const p of PREFIXES) {
			if (rest.startsWith(p) && EVENTS.includes(rest.slice(p.length))) return [rest.slice(p.length), p];
		}
		return EVENTS.includes(rest) ? [rest, 'On'] : [null, 'On'];
	};
	for (const key of Object.keys(effect)) {
		if (!key.startsWith('on') || effect[key] === undefined) continue;
		// `onXPriority: 5` is ordering metadata for `onX`. But `onFractionalPriority` and
		// `onModifyPriority` are events in their own right, and may even be constants
		// (Stall's `onFractionalPriority: -0.1`).
		if (typeof effect[key] !== 'function' && META.test(key) && parse(key.slice(2).replace(META, ''))[0]) continue;
		const [ev, pre] = parse(key.slice(2));
		if (!ev) { out.push({ key, unknown: true }); continue; }
		out.push({ key, ev, pre, kind: 'Fn', constant: typeof effect[key] === 'function' ? undefined : effect[key], ...meta(key) });
	}
	const has = key => effect[key] !== undefined;
	if (['Ability', 'Item'].includes(effect.effectType) && has('onStart') && !has('onSwitchIn') && !has('onAnySwitchIn')) {
		out.push({ key: 'onSwitchIn', ev: 'SwitchIn', pre: 'On', kind: 'StartAlias', ...meta('onSwitchIn') });
	}
	if ((effect.duration || effect.durationCallback) && !has('onResidual')) {
		out.push({ key: 'onResidual', ev: 'Residual', pre: 'On', kind: 'DurationOnly', ...meta('onResidual') });
	}
	return out;
}

/** Every ability the generated Rust table contains, in table order. */
function tableAbilities() {
	const ids = new Set(dex.abilities.all().filter(a => a.exists && !a.isNonstandard).map(a => a.id));
	for (const s of dex.species.all()) {
		if (!s.exists || s.isNonstandard) continue;
		for (const k in s.abilities) ids.add(PS.toID(s.abilities[k]));
	}
	ids.add('noability');
	return [...ids].sort().map(id => dex.abilities.get(id));
}

/** Abilities some Champions species (Megas included) can have. */
// (SUPPORTED_ABILITIES is filled in below, once legalAbilities exists.)
function legalAbilities() {
	const ids = new Set();
	for (const s of dex.species.all()) {
		if (!s.exists || s.isNonstandard) continue;
		for (const k in s.abilities) ids.add(PS.toID(s.abilities[k]));
	}
	return [...ids].sort();
}

/** Every held item the generated Rust table contains, in table order (index 0 is "no item"). */
function tableItems() {
	return dex.items.all().filter(i => i.exists && !i.isNonstandard).sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
}

for (const id of legalAbilities()) if (!DEFERRED_ABILITIES[id]) SUPPORTED_ABILITIES.add(id);
if (process.env.VGC_ABILITIES !== undefined) {
	// Debugging aid: model only the listed abilities.
	const only = new Set(process.env.VGC_ABILITIES.split(',').filter(Boolean));
	for (const id of [...SUPPORTED_ABILITIES]) if (id !== 'noability' && !only.has(id)) SUPPORTED_ABILITIES.delete(id);
}

module.exports = {
	PS, dex, MOD, FORMAT, STAT_IDS, BOOST_IDS, STATUSES, SPECIAL, VOLATILES, MOVE_VOLATILES, EVENTS,
	SUPPORTED_ABILITIES, SUPPORTED_ITEMS, DEFERRED_ABILITIES, DEFERRED_ITEMS, DORMANT_PARTS,
	unsupportedReasons, legalSpecies, learnableMoves, tableMoves, tableSpecies, mulberry32,
	callbacks, tableAbilities, legalAbilities, tableItems,
};
