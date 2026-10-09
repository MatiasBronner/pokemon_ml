//! Event callbacks of abilities, ported from Showdown's `data/abilities.ts`
//! with the Champions overrides.
//!
//! Each arm is one Showdown callback, named in the comment above it together
//! with its parameter list. Inside a callback `self.event` holds what Showdown
//! passes as arguments (the first Pokémon parameter is `e.target`, the second
//! `e.source`, the effect is `e.effect`, a numeric relay value is `relay`),
//! and `holder` is the Pokémon with the ability (`this.effectState.target`).
//! Callbacks that only write to Showdown's battle log are empty here.

#![allow(clippy::collapsible_if, clippy::needless_range_loop)]

use crate::battle::{div1, modify};
use crate::data::*;
use crate::state::*;

/// Moves whose type the "-ate" abilities leave alone.
const NO_MODIFY_TYPE: [&str; 7] =
    ["judgment", "multiattack", "naturalgift", "revelationdance", "technoblast", "terrainpulse", "weatherball"];

/// The volatiles Aroma Veil blocks. None is modelled yet; matching by name
/// means the ability starts working the moment one is.
const AROMA_VEIL: [&str; 6] = ["attract", "disable", "encore", "healblock", "taunt", "torment"];

/// The type a pinch ability (Blaze and friends) boosts at low HP.
fn pinch_type(ability: u16) -> Option<Type> {
    Some(match ability {
        ab::BLAZE => Type::Fire,
        ab::OVERGROW => Type::Grass,
        ab::TORRENT => Type::Water,
        ab::SWARM => Type::Bug,
        _ => return None,
    })
}

/// The type an "-ate" ability turns Normal moves into.
fn ate_type(ability: u16) -> Option<Type> {
    Some(match ability {
        ab::AERILATE => Type::Flying,
        ab::PIXILATE => Type::Fairy,
        ab::REFRIGERATE => Type::Ice,
        ab::DRAGONIZE => Type::Dragon,
        _ => return None,
    })
}

/// The status an immunity ability blocks and cures (`onSetStatus` + `onUpdate`).
fn status_guard(ability: u16) -> Option<&'static [Status]> {
    Some(match ability {
        ab::IMMUNITY => &[Status::Psn, Status::Tox],
        ab::INSOMNIA | ab::VITALSPIRIT => &[Status::Slp],
        ab::LIMBER => &[Status::Par],
        ab::THERMALEXCHANGE | ab::WATERBUBBLE => &[Status::Brn],
        _ => return None,
    })
}

/// Abilities that heal a quarter of max HP instead of being hit by one type.
fn absorb_heal(ability: u16) -> Option<Type> {
    Some(match ability {
        ab::EARTHEATER => Type::Ground,
        ab::VOLTABSORB => Type::Electric,
        ab::WATERABSORB => Type::Water,
        _ => return None,
    })
}

/// Abilities that raise a stat instead of being hit by one type: (type, stat).
fn absorb_boost(ability: u16) -> Option<(Type, usize)> {
    Some(match ability {
        ab::LIGHTNINGROD => (Type::Electric, SPA),
        ab::MOTORDRIVE => (Type::Electric, SPE),
        ab::SAPSIPPER => (Type::Grass, ATK),
        _ => return None,
    })
}

/// Contact abilities that may inflict a status on the attacker.
fn contact_status(ability: u16) -> Option<Status> {
    Some(match ability {
        ab::FLAMEBODY => Status::Brn,
        ab::POISONPOINT => Status::Psn,
        ab::STATIC => Status::Par,
        _ => return None,
    })
}

/// Base-power boosts keyed on a move flag: (flag, numerator over 4096).
fn flag_boost(ability: u16) -> Option<(u32, u32)> {
    Some(match ability {
        ab::IRONFIST => (F_PUNCH, 4915),
        ab::MEGALAUNCHER => (F_PULSE, 6144),
        ab::SHARPNESS => (F_SLICING, 6144),
        ab::STRONGJAW => (F_BITE, 6144),
        ab::TOUGHCLAWS => (F_CONTACT, 5325),
        _ => return None,
    })
}

impl Battle {
    fn clear_negative(&mut self) {
        for b in self.event.boosts.iter_mut() {
            if *b < 0 {
                *b = 0;
            }
        }
    }

    /// The shared `onTryBoost` of the abilities Intimidate cannot touch.
    fn block_intimidate(&mut self) -> Res {
        if self.event.effect == Eff::Ability(ab::INTIMIDATE) && self.event.boosts[ATK] != 0 {
            self.event.boosts[ATK] = 0;
        }
        Res::Undef
    }

    /// `Pokemon#getBestStat(true, true)`: index 1..=5 of the highest raw stat.
    fn best_stat(&self, r: MonRef) -> usize {
        let m = self.mon(r);
        let mut best = 1;
        let mut value = 0;
        for k in 1..=5 {
            if m.stats[k] > value {
                best = k;
                value = m.stats[k];
            }
        }
        best
    }

    /// Runs one ability callback.
    pub(crate) fn ability_cb(&mut self, ability: u16, ev: Ev, pre: Pre, holder: MonRef) -> Res {
        let e = self.event;
        let relay = e.relay;
        let mi = self.event_move();
        let mtype = mi.map(|m| self.am[m as usize].typ);
        let mflags = mi.map_or(0, |m| self.am[m as usize].flags);
        let mcat = mi.map(|m| self.am[m as usize].category);

        // ---- families of abilities that share a callback -----------------------
        if let Some(t) = pinch_type(ability) {
            // onModifyAtk / onModifySpA(atk, attacker, defender, move)
            let m = self.mon(holder);
            if mtype == Some(t) && m.hp as u32 * 3 <= m.max_hp() as u32 {
                return self.chain_modify(6144, 4096);
            }
            return Res::Undef;
        }
        if let Some(t) = ate_type(ability) {
            let Some(m) = mi else {
                return Res::Undef;
            };
            return match ev {
                // onModifyType(move, pokemon)
                Ev::ModifyType => {
                    let am = &mut self.am[m as usize];
                    if am.typ == Type::Normal && !NO_MODIFY_TYPE.contains(&am.d().id) {
                        am.typ = t;
                        am.type_changer_boosted = Eff::Ability(ability);
                    }
                    Res::Undef
                }
                // onBasePower(basePower, pokemon, target, move)
                Ev::BasePower => {
                    if self.am[m as usize].type_changer_boosted == Eff::Ability(ability) {
                        return self.chain_modify(4915, 4096);
                    }
                    Res::Undef
                }
                _ => unreachable!("no body for {} {ev:?}", ABILITIES[ability as usize].id),
            };
        }
        if let (Some(guarded), Ev::Update | Ev::SetStatus) = (status_guard(ability), ev) {
            return match ev {
                // onUpdate(pokemon)
                Ev::Update => {
                    if guarded.contains(&self.mon(holder).status) {
                        self.cure_status(holder);
                    }
                    Res::Undef
                }
                // onSetStatus(status, target, source, effect)
                _ => {
                    if guarded.contains(&e.status) {
                        return FALSE;
                    }
                    Res::Undef
                }
            };
        }
        if let (Some(t), Ev::TryHit, Pre::On) = (absorb_heal(ability), ev, pre) {
            // onTryHit(target, source, move)
            if e.target != e.source && mtype == Some(t) {
                let amount = div1(self.mon(holder).max_hp() as u32, 4);
                self.heal(amount, None, None, Eff::None);
                return Res::Null;
            }
            return Res::Undef;
        }
        if let (Some((t, stat)), Ev::TryHit, Pre::On) = (absorb_boost(ability), ev, pre) {
            // onTryHit(target, source, move)
            if e.target != e.source && mtype == Some(t) {
                self.boost1(stat, 1, None, None, Eff::None);
                return Res::Null;
            }
            return Res::Undef;
        }
        if let (Some(status), Ev::DamagingHit) = (contact_status(ability), ev) {
            // onDamagingHit(damage, target, source, move)
            if let (Some(m), Some(source)) = (mi, e.source) {
                if self.makes_contact(m) && self.chance(3, 10, "contact ability") {
                    self.try_set_status(source, status, e.target, Eff::None);
                }
            }
            return Res::Undef;
        }
        if let (Some((flag, num)), Ev::BasePower) = (flag_boost(ability), ev) {
            // onBasePower(basePower, attacker, defender, move)
            if mflags & flag != 0 {
                return self.chain_modify(num, 4096);
            }
            return Res::Undef;
        }

        match (ability, ev, pre) {
            // ---- Adaptability: onModifySTAB(stab, source, target, move)
            (ab::ADAPTABILITY, Ev::ModifySTAB, Pre::On) => {
                if let (Some(t), Some(source)) = (mtype, e.target) {
                    if self.has_type(source, t) {
                        return Res::Num(if relay == Res::Num(8192) { 9216 } else { 8192 });
                    }
                }
                Res::Undef
            }

            // ---- Aftermath: onDamagingHit(damage, target, source, move)
            (ab::AFTERMATH, Ev::DamagingHit, Pre::On) => {
                if let (Some(m), Some(target), Some(source)) = (mi, e.target, e.source) {
                    if self.mon(target).hp == 0 && self.makes_contact(m) {
                        let d = div1(self.mon(source).max_hp() as u32, 4);
                        self.damage(d, Some(source), Some(target), Eff::None);
                    }
                }
                Res::Undef
            }

            // ---- Analytic: onBasePower(basePower, pokemon)
            (ab::ANALYTIC, Ev::BasePower, Pre::On) => {
                let (actives, n) = self.all_active(false);
                let boosted = !actives[..n].iter().any(|&t| Some(t) != e.target && self.will_move(t));
                if boosted {
                    return self.chain_modify(5325, 4096);
                }
                Res::Undef
            }

            // ---- Anger Point: onHit(target, source, move)
            (ab::ANGERPOINT, Ev::Hit, Pre::On) => {
                if let (Some(m), Some(target)) = (mi, e.target) {
                    if self.mon(target).hp > 0 && self.am[m as usize].hit_data[self.slot_index(target)].crit {
                        self.boost1(ATK, 12, Some(target), Some(target), Eff::None);
                    }
                }
                Res::Undef
            }

            // ---- Anticipation, Frisk, Pressure, Mold Breaker, Fairy Aura: onStart only announces.
            (ab::ANTICIPATION | ab::FRISK | ab::PRESSURE | ab::MOLDBREAKER | ab::FAIRYAURA, Ev::Start, Pre::On) => {
                Res::Undef
            }
            // ---- Screen Cleaner: onStart removes screens, of which there are none yet.
            (ab::SCREENCLEANER, Ev::Start, Pre::On) => Res::Undef,
            // ---- Embody Aspect: onStart only acts on a Terastallized Ogerpon; Champions has no Terastallization.
            (
                ab::EMBODYASPECTCORNERSTONE
                | ab::EMBODYASPECTHEARTHFLAME
                | ab::EMBODYASPECTTEAL
                | ab::EMBODYASPECTWELLSPRING,
                Ev::Start,
                Pre::On,
            ) => Res::Undef,
            // ---- Gluttony only matters for berries that are not in Champions.
            (ab::GLUTTONY, Ev::Start | Ev::Damage, Pre::On) => Res::Undef,
            // ---- Heavy Metal / Light Metal: nothing modelled asks for a weight.
            (ab::HEAVYMETAL | ab::LIGHTMETAL, Ev::ModifyWeight, Pre::On) => Res::Undef,
            // ---- Suction Cups: onDragOut
            (ab::SUCTIONCUPS, Ev::DragOut, Pre::On) => Res::Null,

            // ---- Armor Tail / Queenly Majesty: onFoeTryMove(target, source, move).
            // Here `target` is the Pokémon using the move and `source` its target.
            (ab::ARMORTAIL | ab::QUEENLYMAJESTY, Ev::TryMove, Pre::Foe) => {
                let (Some(m), Some(aimed_at)) = (mi, e.source) else {
                    return Res::Undef;
                };
                if self.is_ally(aimed_at, holder) && self.am[m as usize].priority > 0 {
                    return FALSE;
                }
                Res::Undef
            }

            // ---- Aroma Veil: onAllyTryAddVolatile(status, target, source, effect)
            (ab::AROMAVEIL, Ev::TryAddVolatile, Pre::Ally) => {
                if e.vol.is_some_and(|v| AROMA_VEIL.contains(&v.id())) {
                    return Res::Null;
                }
                Res::Undef
            }

            // ---- Aura Guard: onSourceModifyDamage(damage, source, target, move)
            (ab::AURAGUARD, Ev::ModifyDamage, Pre::Source) => {
                if mflags & F_CONTACT != 0 {
                    return self.chain_modify(2048, 4096);
                }
                Res::Undef
            }

            // ---- Battle Armor / Shell Armor: onCriticalHit: false
            (ab::BATTLEARMOR | ab::SHELLARMOR, Ev::CriticalHit, Pre::On) => FALSE,

            // ---- Berserk (Champions). `ability_st.a` is `checkedBerserk`: 0 unset, 1 true, 2 false.
            // onDamage(damage, target, source, effect)
            (ab::BERSERK, Ev::Damage, Pre::On) => {
                let single_hit_move = match e.effect {
                    Eff::Move(m) => self.am[m as usize].multihit == (0, 0),
                    Eff::Confused => true,
                    _ => false,
                };
                self.mon_mut(holder).ability_st.a = if single_hit_move { 2 } else { 1 };
                Res::Undef
            }
            // onTryEatItem(item)
            (ab::BERSERK, Ev::TryEatItem, Pre::On) => {
                if e.item == it::SITRUSBERRY || e.item == it::ORANBERRY {
                    return match self.mon(holder).ability_st.a {
                        1 => TRUE,
                        2 => FALSE,
                        _ => Res::Undef,
                    };
                }
                TRUE
            }
            // onAfterMoveSecondary(target, source, move)
            (ab::BERSERK, Ev::AfterMoveSecondary, Pre::On) => {
                self.mon_mut(holder).ability_st.a = 1;
                let (Some(m), Some(target), Some(source)) = (mi, e.target, e.source) else {
                    return Res::Undef;
                };
                let am = &self.am[m as usize];
                let t = self.mon(target);
                if source == target || t.hp == 0 || am.total_damage == 0 || !t.was_attacked {
                    return Res::Undef;
                }
                let damage = if am.multihit != (0, 0) { am.total_damage } else { t.last_attack_damage };
                let (hp, max) = (t.hp as i32, t.max_hp() as i32);
                if hp * 2 <= max && (hp + damage) * 2 > max {
                    self.boost1(SPA, 1, Some(target), Some(target), Eff::None);
                }
                Res::Undef
            }

            // ---- Big Pecks / Hyper Cutter / Keen Eye / Illuminate: onTryBoost(boost, target, source, effect)
            (ab::BIGPECKS | ab::HYPERCUTTER | ab::KEENEYE | ab::ILLUMINATE, Ev::TryBoost, Pre::On) => {
                if e.source.is_some() && e.target == e.source {
                    return Res::Undef;
                }
                let stat = match ability {
                    ab::BIGPECKS => DEF,
                    ab::HYPERCUTTER => ATK,
                    _ => ACC,
                };
                if self.event.boosts[stat] < 0 {
                    self.event.boosts[stat] = 0;
                }
                Res::Undef
            }
            // ---- Keen Eye / Illuminate: onModifyMove(move)
            (ab::KEENEYE | ab::ILLUMINATE, Ev::ModifyMove, Pre::On) => {
                if let Some(m) = mi {
                    self.am[m as usize].ignore_evasion = true;
                }
                Res::Undef
            }

            // ---- Bulletproof: onTryHit(pokemon, target, move)
            (ab::BULLETPROOF, Ev::TryHit, Pre::On) => {
                if mflags & F_BULLET != 0 {
                    return Res::Null;
                }
                Res::Undef
            }

            // ---- Cheek Pouch: onEatItem(item, pokemon)
            (ab::CHEEKPOUCH, Ev::EatItem, Pre::On) => {
                let amount = div1(self.mon(holder).max_hp() as u32, 3);
                self.heal(amount, None, None, Eff::None);
                Res::Undef
            }

            // ---- Clear Body / White Smoke: onTryBoost(boost, target, source, effect)
            (ab::CLEARBODY | ab::WHITESMOKE, Ev::TryBoost, Pre::On) => {
                if e.source.is_some() && e.target == e.source {
                    return Res::Undef;
                }
                self.clear_negative();
                Res::Undef
            }

            // ---- Competitive / Defiant: onAfterEachBoost(boost, target, source, effect)
            (ab::COMPETITIVE | ab::DEFIANT, Ev::AfterEachBoost, Pre::On) => {
                let (Some(target), Some(source)) = (e.target, e.source) else {
                    return Res::Undef;
                };
                if self.is_ally(target, source) {
                    return Res::Undef;
                }
                if e.boosts.iter().any(|&b| b < 0) {
                    let stat = if ability == ab::DEFIANT { ATK } else { SPA };
                    self.boost1(stat, 2, Some(target), Some(target), Eff::None);
                }
                Res::Undef
            }

            // ---- Compound Eyes: onSourceModifyAccuracy(accuracy)
            (ab::COMPOUNDEYES, Ev::ModifyAccuracy, Pre::Source) => {
                if matches!(relay, Res::Num(_)) {
                    return self.chain_modify(5325, 4096);
                }
                Res::Undef
            }

            // ---- Contrary: onChangeBoost(boost, target, source, effect)
            (ab::CONTRARY, Ev::ChangeBoost, Pre::On) => {
                for b in self.event.boosts.iter_mut() {
                    *b = -*b;
                }
                Res::Undef
            }

            // ---- Cud Chew. `ability_st.a` is the berry to eat again, `.b` the turns left.
            // onEatItem(item, pokemon, source, effect)
            (ab::CUDCHEW, Ev::EatItem, Pre::On) => {
                if ITEMS[e.item as usize].flags & IF_BERRY != 0 {
                    let last_action = self.queue.peek().is_none();
                    let st = &mut self.mon_mut(holder).ability_st;
                    st.a = e.item as i16;
                    st.b = if last_action { 1 } else { 2 };
                }
                Res::Undef
            }
            // onResidual(pokemon)
            (ab::CUDCHEW, Ev::Residual, Pre::On) => {
                let st = self.mon(holder).ability_st;
                if st.a == 0 || self.mon(holder).hp == 0 {
                    return Res::Undef;
                }
                self.mon_mut(holder).ability_st.b -= 1;
                if self.mon(holder).ability_st.b <= 0 {
                    let item = st.a as u16;
                    if self
                        .single_event(Ev::Eat, Eff::Item(item), None, Some(holder), None, Eff::None, Res::Undef)
                        .truthy()
                    {
                        let mut ee = Event::new(Ev::EatItem, Some(holder), None, Eff::None);
                        ee.item = item;
                        self.run_event_ex(ee, Res::Undef, false, false);
                    }
                    let m = self.mon_mut(holder);
                    m.ate_berry = true;
                    m.ability_st.a = 0;
                    m.ability_st.b = 0;
                }
                Res::Undef
            }

            // ---- Curious Medicine: onStart(pokemon)
            (ab::CURIOUSMEDICINE, Ev::Start, Pre::On) => {
                let (allies, n) = self.adjacent_allies(holder);
                for &a in &allies[..n] {
                    self.mon_mut(a).boosts = [0; 7];
                }
                Res::Undef
            }

            // ---- Damp
            // onAnyTryMove(target, source, effect): the self-destructing moves are not modelled.
            (ab::DAMP, Ev::TryMove, Pre::Any) => Res::Undef,
            // onAnyDamage(damage, target, source, effect)
            (ab::DAMP, Ev::Damage, Pre::Any) => {
                if e.effect == Eff::Ability(ab::AFTERMATH) {
                    return FALSE;
                }
                Res::Undef
            }

            // ---- Eelevate: onSourceAfterFaint(length, target, source, effect)
            (ab::EELEVATE, Ev::AfterFaint, Pre::Source) => {
                if let (true, Some(source)) = (e.effect.is_move(), e.source) {
                    let stat = self.best_stat(source) - 1;
                    self.boost1(stat, relay.num() as i8, Some(source), None, Eff::None);
                }
                Res::Undef
            }

            // ---- Effect Spore: onDamagingHit(damage, target, source, move)
            (ab::EFFECTSPORE, Ev::DamagingHit, Pre::On) => {
                let (Some(m), Some(source)) = (mi, e.source) else {
                    return Res::Undef;
                };
                if self.makes_contact(m) && self.run_status_immunity(source, Imm::Powder) {
                    let r = self.rand(100, "effect spore");
                    let status = if r < 11 {
                        Status::Slp
                    } else if r < 21 {
                        Status::Par
                    } else if r < 30 {
                        Status::Psn
                    } else {
                        Status::None
                    };
                    if status != Status::None {
                        self.try_set_status(source, status, e.target, Eff::None);
                    }
                }
                Res::Undef
            }

            // ---- Fairy Aura: onAnyBasePower(basePower, source, target, move)
            (ab::FAIRYAURA, Ev::BasePower, Pre::Any) => {
                let Some(m) = mi else {
                    return Res::Undef;
                };
                if e.target == e.source || mcat == Some(Category::Status) || mtype != Some(Type::Fairy) {
                    return Res::Undef;
                }
                let booster = self.am[m as usize].aura_booster;
                if !booster.is_some_and(|b| self.has_ability(b, ab::FAIRYAURA)) {
                    self.am[m as usize].aura_booster = Some(holder);
                }
                if self.am[m as usize].aura_booster != Some(holder) {
                    return Res::Undef;
                }
                self.chain_modify(5448, 4096)
            }

            // ---- Filter / Solid Rock: onSourceModifyDamage(damage, source, target, move)
            (ab::FILTER | ab::SOLIDROCK, Ev::ModifyDamage, Pre::Source) => {
                if let (Some(m), Some(target)) = (mi, e.source) {
                    if self.am[m as usize].hit_data[self.slot_index(target)].type_mod > 0 {
                        return self.chain_modify(3072, 4096);
                    }
                }
                Res::Undef
            }

            // ---- Fire Mane: onModifyAtk / onModifySpA(atk, attacker, defender, move)
            (ab::FIREMANE, Ev::ModifyAtk | Ev::ModifySpA, Pre::On) => {
                if mtype == Some(Type::Fire) {
                    return self.chain_modify(6144, 4096);
                }
                Res::Undef
            }

            // ---- Flash Fire
            // onTryHit(target, source, move)
            (ab::FLASHFIRE, Ev::TryHit, Pre::On) => {
                if let (Some(m), true) = (mi, e.target != e.source && mtype == Some(Type::Fire)) {
                    self.am[m as usize].accuracy = 0;
                    self.add_volatile(holder, VolKind::Flashfire, None, Eff::None);
                    return Res::Null;
                }
                Res::Undef
            }
            // onEnd(pokemon)
            (ab::FLASHFIRE, Ev::End, Pre::On) => {
                self.remove_volatile(holder, VolKind::Flashfire);
                Res::Undef
            }

            // ---- Flower Veil
            // onAllyTryBoost(boost, target, source, effect)
            (ab::FLOWERVEIL, Ev::TryBoost, Pre::Ally) => {
                let Some(target) = e.target else {
                    return Res::Undef;
                };
                if (e.source.is_some() && e.target == e.source) || !self.has_type(target, Type::Grass) {
                    return Res::Undef;
                }
                self.clear_negative();
                Res::Undef
            }
            // onAllySetStatus(status, target, source, effect)
            (ab::FLOWERVEIL, Ev::SetStatus, Pre::Ally) => {
                let Some(target) = e.target else {
                    return Res::Undef;
                };
                if self.has_type(target, Type::Grass) && e.source.is_some() && e.target != e.source {
                    return Res::Null;
                }
                Res::Undef
            }
            // onAllyTryAddVolatile(status, target): only blocks Yawn, which is not modelled.
            (ab::FLOWERVEIL | ab::SWEETVEIL, Ev::TryAddVolatile, Pre::Ally) => Res::Undef,

            // ---- Fluffy: onSourceModifyDamage(damage, source, target, move)
            (ab::FLUFFY, Ev::ModifyDamage, Pre::Source) => {
                let mut num = 4096;
                if mtype == Some(Type::Fire) {
                    num *= 2;
                }
                if mflags & F_CONTACT != 0 {
                    num /= 2;
                }
                self.chain_modify(num, 4096)
            }

            // ---- Forewarn: onStart(pokemon). Showdown picks at random among the
            // foes' strongest moves; only the draw matters here.
            (ab::FOREWARN, Ev::Start, Pre::On) => {
                let (foes, n) = self.allies_and_self(1 - holder.side as usize);
                let mut best = 1;
                let mut count = 0;
                for &f in &foes[..n] {
                    let m = self.mon(f);
                    for slot in &m.moves[..m.n_moves as usize] {
                        let d = &MOVES[slot.id as usize];
                        let mut bp = d.base_power as u32;
                        if bp == 1 || (bp == 0 && d.category != Category::Status) {
                            bp = 80;
                        }
                        if bp > best {
                            best = bp;
                            count = 1;
                        } else if bp == best {
                            count += 1;
                        }
                    }
                }
                if count > 0 {
                    self.rand(count, "forewarn");
                }
                Res::Undef
            }

            // ---- Friend Guard: onAnyModifyDamage(damage, source, target, move)
            (ab::FRIENDGUARD, Ev::ModifyDamage, Pre::Any) => {
                if let Some(target) = e.source {
                    if target != holder && self.is_ally(target, holder) {
                        return self.chain_modify(3072, 4096);
                    }
                }
                Res::Undef
            }

            // ---- Fur Coat: onModifyDef(def)
            (ab::FURCOAT, Ev::ModifyDef, Pre::On) => self.chain_modify(2, 1),

            // ---- Gale Wings: onModifyPriority(priority, pokemon, target, move)
            (ab::GALEWINGS, Ev::ModifyPriority, Pre::On) => {
                let m = self.mon(holder);
                if mtype == Some(Type::Flying) && m.hp == m.max_hp() {
                    return Res::Num(relay.num() + 1);
                }
                Res::Undef
            }

            // ---- Good as Gold: onTryHit(target, source, move)
            (ab::GOODASGOLD, Ev::TryHit, Pre::On) => {
                if mcat == Some(Category::Status) && e.target != e.source {
                    return Res::Null;
                }
                Res::Undef
            }

            // ---- Gooey: onDamagingHit(damage, target, source, move)
            (ab::GOOEY, Ev::DamagingHit, Pre::On) => {
                if let (Some(m), Some(source)) = (mi, e.source) {
                    if self.makes_contact(m) {
                        self.boost1(SPE, -1, Some(source), e.target, Eff::None);
                    }
                }
                Res::Undef
            }

            // ---- Guard Dog
            (ab::GUARDDOG, Ev::DragOut, Pre::On) => Res::Null,
            // onTryBoost(boost, target, source, effect)
            (ab::GUARDDOG, Ev::TryBoost, Pre::On) => {
                if e.effect == Eff::Ability(ab::INTIMIDATE) && self.event.boosts[ATK] != 0 {
                    self.event.boosts[ATK] = 0;
                    self.boost1(ATK, 1, e.target, e.target, Eff::None);
                }
                Res::Undef
            }

            // ---- Guts: onModifyAtk(atk, pokemon)
            (ab::GUTS, Ev::ModifyAtk, Pre::On) => {
                if self.mon(holder).status != Status::None {
                    return self.chain_modify(6144, 4096);
                }
                Res::Undef
            }

            // ---- Harvest: onResidual(pokemon). (In sun it always works; there is no weather yet.)
            (ab::HARVEST, Ev::Residual, Pre::On) => {
                if self.chance(1, 2, "harvest") {
                    let m = self.mon(holder);
                    if m.hp > 0 && m.item == it::NONE && ITEMS[m.last_item as usize].flags & IF_BERRY != 0 {
                        let item = m.last_item;
                        self.set_item(holder, item, None, Eff::None);
                        self.mon_mut(holder).last_item = it::NONE;
                    }
                }
                Res::Undef
            }

            // ---- Healer (Champions: one chance in two): onResidual(pokemon)
            (ab::HEALER, Ev::Residual, Pre::On) => {
                let (allies, n) = self.adjacent_allies(holder);
                for &a in &allies[..n] {
                    if self.mon(a).status != Status::None && self.chance(1, 2, "healer") {
                        self.cure_status(a);
                    }
                }
                Res::Undef
            }

            // ---- Heatproof
            // onSourceModifyAtk / onSourceModifySpA(atk, attacker, defender, move)
            (ab::HEATPROOF, Ev::ModifyAtk | Ev::ModifySpA, Pre::Source) => {
                if mtype == Some(Type::Fire) {
                    return self.chain_modify(2048, 4096);
                }
                Res::Undef
            }
            // onDamage(damage, target, source, effect): burn damage is halved.
            (ab::HEATPROOF, Ev::Damage, Pre::On) => {
                if e.effect == Eff::Status(Status::Brn) {
                    return Res::Num(relay.num() / 2);
                }
                Res::Undef
            }

            // ---- Hospitality: onStart(pokemon)
            (ab::HOSPITALITY, Ev::Start, Pre::On) => {
                let (allies, n) = self.adjacent_allies(holder);
                for &a in &allies[..n] {
                    let amount = div1(self.mon(a).max_hp() as u32, 4);
                    self.heal(amount, Some(a), Some(holder), Eff::None);
                }
                Res::Undef
            }

            // ---- Huge Power / Pure Power: onModifyAtk(atk)
            (ab::HUGEPOWER | ab::PUREPOWER, Ev::ModifyAtk, Pre::On) => self.chain_modify(2, 1),

            // ---- Hustle
            // onModifyAtk(atk): applied to the stat directly, not chained.
            (ab::HUSTLE, Ev::ModifyAtk, Pre::On) => Res::Num(modify(relay.num() as u32, 6144) as i32),
            // onSourceModifyAccuracy(accuracy, target, source, move)
            (ab::HUSTLE, Ev::ModifyAccuracy, Pre::Source) => {
                if mcat == Some(Category::Physical) && matches!(relay, Res::Num(_)) {
                    return self.chain_modify(3277, 4096);
                }
                Res::Undef
            }

            // ---- Infiltrator: onModifyMove(move)
            (ab::INFILTRATOR, Ev::ModifyMove, Pre::On) => {
                if let Some(m) = mi {
                    self.am[m as usize].infiltrates = true;
                }
                Res::Undef
            }

            // ---- Innards Out: onDamagingHit(damage, target, source, move)
            (ab::INNARDSOUT, Ev::DamagingHit, Pre::On) => {
                if let (Some(m), Some(target), Some(source)) = (mi, e.target, e.source) {
                    if self.mon(target).hp == 0 {
                        let d = relay.num() + self.am[m as usize].total_damage;
                        self.damage(d, Some(source), Some(target), Eff::None);
                    }
                }
                Res::Undef
            }

            // ---- Inner Focus
            // onTryAddVolatile(status, pokemon)
            (ab::INNERFOCUS, Ev::TryAddVolatile, Pre::On) => {
                if e.vol == Some(VolKind::Flinch) {
                    return Res::Null;
                }
                Res::Undef
            }
            // onTryBoost(boost, target, source, effect)
            (ab::INNERFOCUS | ab::OBLIVIOUS | ab::OWNTEMPO | ab::SCRAPPY, Ev::TryBoost, Pre::On) => {
                self.block_intimidate()
            }

            // ---- Insomnia / Vital Spirit / Purifying Salt: onTryAddVolatile only blocks Yawn, which is not modelled.
            (ab::INSOMNIA | ab::VITALSPIRIT | ab::PURIFYINGSALT, Ev::TryAddVolatile, Pre::On) => Res::Undef,

            // ---- Intimidate: onStart(pokemon)
            (ab::INTIMIDATE, Ev::Start, Pre::On) => {
                let (foes, n) = self.allies_and_self(1 - holder.side as usize);
                for &f in &foes[..n] {
                    self.boost1(ATK, -1, Some(f), Some(holder), Eff::None);
                }
                Res::Undef
            }

            // ---- Justified: onDamagingHit(damage, target, source, move)
            (ab::JUSTIFIED, Ev::DamagingHit, Pre::On) => {
                if mtype == Some(Type::Dark) {
                    self.boost1(ATK, 1, None, None, Eff::None);
                }
                Res::Undef
            }

            // ---- Klutz: onStart(pokemon) ends the held item's effect.
            (ab::KLUTZ, Ev::Start, Pre::On) => {
                let item = self.mon(holder).item;
                self.single_event(Ev::End, Eff::Item(item), Some(holder), Some(holder), None, Eff::None, Res::Undef);
                Res::Undef
            }

            // ---- Libero / Protean: onPrepareHit(source, target, move). Once per stay on the field.
            (ab::LIBERO | ab::PROTEAN, Ev::PrepareHit, Pre::On) => {
                let Some(m) = mi else {
                    return Res::Undef;
                };
                if self.mon(holder).ability_st.a != 0 || self.am[m as usize].has_bounced {
                    return Res::Undef;
                }
                let t = self.am[m as usize].typ;
                if t != Type::Typeless && t != Type::None && self.mon(holder).types != [t, Type::None] {
                    let mon = self.mon_mut(holder);
                    mon.types = [t, Type::None];
                    mon.ability_st.a = 1;
                }
                Res::Undef
            }

            // ---- Lightning Rod: onAnyRedirectTarget(target, source, source2, move)
            (ab::LIGHTNINGROD, Ev::RedirectTarget, Pre::Any) => {
                let (Some(m), Some(user)) = (mi, e.target) else {
                    return Res::Undef;
                };
                if mtype != Some(Type::Electric) {
                    return Res::Undef;
                }
                let t = match self.am[m as usize].target {
                    Target::RandomNormal | Target::AdjacentFoe => Target::Normal,
                    other => other,
                };
                let loc = self.loc_of(user, holder);
                if self.valid_target_loc(loc, user, t) {
                    return Res::Mon(holder);
                }
                Res::Undef
            }

            // ---- Liquid Ooze: onSourceTryHeal(damage, target, source, effect)
            (ab::LIQUIDOOZE, Ev::TryHeal, Pre::Source) => {
                if e.effect == Eff::Drain {
                    self.damage(relay.num(), None, None, Eff::None);
                    return Res::Num(0);
                }
                Res::Undef
            }

            // ---- Liquid Voice: onModifyType(move, pokemon)
            (ab::LIQUIDVOICE, Ev::ModifyType, Pre::On) => {
                if let (Some(m), true) = (mi, mflags & F_SOUND != 0) {
                    self.am[m as usize].typ = Type::Water;
                }
                Res::Undef
            }

            // ---- Long Reach: onModifyMove(move)
            (ab::LONGREACH, Ev::ModifyMove, Pre::On) => {
                if let Some(m) = mi {
                    self.am[m as usize].flags &= !F_CONTACT;
                }
                Res::Undef
            }

            // ---- Magic Bounce: onTryHit(target, source, move)
            (ab::MAGICBOUNCE, Ev::TryHit, Pre::On) => {
                let (Some(m), Some(target), Some(source)) = (mi, e.target, e.source) else {
                    return Res::Undef;
                };
                let am = &self.am[m as usize];
                if target == source || am.has_bounced || am.flags & F_REFLECTABLE == 0 {
                    return Res::Undef;
                }
                let id = am.id;
                let bounced = self.new_am(id);
                self.am[bounced as usize].has_bounced = true;
                self.am[bounced as usize].prankster_boosted = false;
                self.use_move(bounced, target, Some(source), Eff::None);
                Res::Null
            }
            // onAllyTryHitSide: nothing modelled targets a side.
            (ab::MAGICBOUNCE | ab::SAPSIPPER | ab::SOUNDPROOF, Ev::TryHitSide, Pre::Ally) => Res::Undef,

            // ---- Magic Guard: onDamage(damage, target, source, effect)
            (ab::MAGICGUARD, Ev::Damage, Pre::On) => {
                if !e.effect.is_move() {
                    return FALSE;
                }
                Res::Undef
            }

            // ---- Magician: onAfterMoveSecondarySelf(source, target, move)
            (ab::MAGICIAN, Ev::AfterMoveSecondarySelf, Pre::On) => {
                let (Some(m), Some(source)) = (mi, e.target) else {
                    return Res::Undef;
                };
                let am = &self.am[m as usize];
                let s = self.mon(source);
                if !am.has_hit_targets
                    || s.item != it::NONE
                    || s.volatiles.has(VolKind::Gem)
                    || am.category == Category::Status
                {
                    return Res::Undef;
                }
                // Showdown speed-sorts the hit targets in place, ties shuffled.
                let n = am.n_hit_targets as usize;
                let mut keyed = [(source, 0i32); 3];
                for i in 0..n {
                    keyed[i] = (am.hit_targets[i], self.mon(am.hit_targets[i]).speed);
                }
                self.speed_sort(&mut keyed[..n], |a, b| b.1 as i64 - a.1 as i64, "magician target tie");
                for i in 0..n {
                    self.am[m as usize].hit_targets[i] = keyed[i].0;
                }
                for &(pokemon, _) in &keyed[..n] {
                    if pokemon == source {
                        continue;
                    }
                    let taken = self.take_item(pokemon, Some(source));
                    if taken == it::NONE {
                        continue;
                    }
                    if !self.set_item(source, taken, None, Eff::None) {
                        // Give it straight back without the usual item hooks.
                        self.mon_mut(pokemon).item = taken;
                        continue;
                    }
                    return Res::Undef;
                }
                Res::Undef
            }

            // ---- Magma Armor
            // onUpdate(pokemon)
            (ab::MAGMAARMOR, Ev::Update, Pre::On) => {
                if self.mon(holder).status == Status::Frz {
                    self.cure_status(holder);
                }
                Res::Undef
            }
            // onImmunity(type, pokemon)
            (ab::MAGMAARMOR, Ev::Immunity, Pre::On) => {
                if e.imm == Some(Imm::Status(Status::Frz)) {
                    return FALSE;
                }
                Res::Undef
            }

            // ---- Marvel Scale: onModifyDef(def, pokemon)
            (ab::MARVELSCALE, Ev::ModifyDef, Pre::On) => {
                if self.mon(holder).status != Status::None {
                    return self.chain_modify(6144, 4096);
                }
                Res::Undef
            }

            // ---- Merciless: onModifyCritRatio(critRatio, source, target)
            (ab::MERCILESS, Ev::ModifyCritRatio, Pre::On) => {
                if e.source.is_some_and(|t| matches!(self.mon(t).status, Status::Psn | Status::Tox)) {
                    return Res::Num(5);
                }
                Res::Undef
            }

            // ---- Minus / Plus: onModifySpA(spa, pokemon)
            (ab::MINUS | ab::PLUS, Ev::ModifySpA, Pre::On) => {
                let (allies, n) = self.allies_and_self(holder.side as usize);
                for &a in &allies[..n] {
                    if a != holder && (self.has_ability(a, ab::MINUS) || self.has_ability(a, ab::PLUS)) {
                        return self.chain_modify(6144, 4096);
                    }
                }
                Res::Undef
            }

            // ---- Mirror Armor: onTryBoost(boost, target, source, effect)
            (ab::MIRRORARMOR, Ev::TryBoost, Pre::On) => {
                let (Some(target), Some(source)) = (e.target, e.source) else {
                    return Res::Undef;
                };
                if target == source || e.effect == Eff::Ability(ab::MIRRORARMOR) {
                    return Res::Undef;
                }
                for k in 0..7 {
                    let b = self.event.boosts[k];
                    if b < 0 {
                        if self.mon(target).boosts[k] == -6 {
                            continue;
                        }
                        self.event.boosts[k] = 0;
                        if self.mon(source).hp > 0 {
                            self.boost1(k, b, Some(source), Some(target), Eff::None);
                        }
                    }
                }
                Res::Undef
            }

            // ---- Mold Breaker: onModifyMove(move)
            (ab::MOLDBREAKER, Ev::ModifyMove, Pre::On) => {
                if let Some(m) = mi {
                    self.am[m as usize].ignore_ability = true;
                }
                Res::Undef
            }

            // ---- Moody: onResidual(pokemon)
            (ab::MOODY, Ev::Residual, Pre::On) => {
                let boosts = self.mon(holder).boosts;
                let mut stats = [0usize; 5];
                let mut n = 0;
                for k in 0..5 {
                    if boosts[k] < 6 {
                        stats[n] = k;
                        n += 1;
                    }
                }
                let plus = if n > 0 { Some(stats[self.rand(n as u32, "moody raise") as usize]) } else { None };
                n = 0;
                for k in 0..5 {
                    if boosts[k] > -6 && Some(k) != plus {
                        stats[n] = k;
                        n += 1;
                    }
                }
                let minus = if n > 0 { Some(stats[self.rand(n as u32, "moody lower") as usize]) } else { None };
                // The raised stat is applied first, whatever its position.
                let mut b = [0i8; 7];
                let mut order = [0u8; 7];
                let mut w = 0;
                for (stat, by) in [(plus, 2), (minus, -1)] {
                    if let Some(k) = stat {
                        b[k] = by;
                        order[w] = k as u8;
                        w += 1;
                    }
                }
                for k in 0..7u8 {
                    if !order[..w].contains(&k) {
                        order[w] = k;
                        w += 1;
                    }
                }
                self.boost_ordered(b, &order, Some(holder), Some(holder), Eff::None);
                Res::Undef
            }

            // ---- Moxie: onSourceAfterFaint(length, target, source, effect)
            (ab::MOXIE, Ev::AfterFaint, Pre::Source) => {
                if e.effect.is_move() {
                    self.boost1(ATK, relay.num() as i8, e.source, None, Eff::None);
                }
                Res::Undef
            }

            // ---- Multiscale: onSourceModifyDamage(damage, source, target, move)
            (ab::MULTISCALE, Ev::ModifyDamage, Pre::Source) => {
                let m = self.mon(holder);
                if m.hp >= m.max_hp() {
                    return self.chain_modify(2048, 4096);
                }
                Res::Undef
            }

            // ---- Mummy: onDamagingHit(damage, target, source, move)
            (ab::MUMMY, Ev::DamagingHit, Pre::On) => {
                let (Some(m), Some(source)) = (mi, e.source) else {
                    return Res::Undef;
                };
                let theirs = self.mon(source).ability;
                if ABILITIES[theirs as usize].flags & AF_CANTSUPPRESS != 0 || theirs == ab::MUMMY {
                    return Res::Undef;
                }
                if self.makes_contact(m) {
                    self.set_ability(source, ab::MUMMY, e.target, Eff::None);
                }
                Res::Undef
            }

            // ---- Natural Cure (Champions): onSwitchOut(pokemon)
            (ab::NATURALCURE, Ev::SwitchOut, Pre::On) => {
                self.cure_status(holder);
                Res::Undef
            }

            // ---- No Guard
            // onAnyInvulnerability(target, source, move)
            (ab::NOGUARD, Ev::Invulnerability, Pre::Any) => {
                if mi.is_some() && (e.source == Some(holder) || e.target == Some(holder)) {
                    return Res::Num(0);
                }
                Res::Undef
            }
            // onAnyAccuracy(accuracy, target, source, move)
            (ab::NOGUARD, Ev::Accuracy, Pre::Any) => {
                if mi.is_some() && (e.source == Some(holder) || e.target == Some(holder)) {
                    return TRUE;
                }
                relay
            }

            // ---- Oblivious: Attract and Taunt are not modelled, so only the Intimidate block does anything.
            (ab::OBLIVIOUS, Ev::Update | Ev::Immunity | Ev::TryHit, Pre::On) => Res::Undef,

            // ---- Opportunist. `ability_boosts` holds the copies waiting to be applied,
            // `ability_st.a` whether Showdown's `effectState.boosts` exists at all.
            // onFoeAfterBoost(boost, target, source, effect)
            (ab::OPPORTUNIST, Ev::AfterBoost, Pre::Foe) => {
                if e.effect == Eff::Ability(ab::OPPORTUNIST) {
                    return Res::Undef;
                }
                let m = self.mon_mut(holder);
                m.ability_st.a = 1;
                for k in 0..7 {
                    if e.boosts[k] > 0 {
                        m.ability_boosts[k] += e.boosts[k];
                    }
                }
                Res::Undef
            }
            // onAnySwitchIn / onAnyAfterMega / onAnyAfterTerastallization / onAnyAfterMove / onResidual
            (ab::OPPORTUNIST, Ev::SwitchIn | Ev::AfterMega | Ev::AfterTerastallization | Ev::AfterMove, Pre::Any)
            | (ab::OPPORTUNIST, Ev::Residual, Pre::On) => {
                if self.mon(holder).ability_st.a == 0 {
                    return Res::Undef;
                }
                let b = self.mon(holder).ability_boosts;
                self.boost(b, Some(holder), None, Eff::None);
                let m = self.mon_mut(holder);
                m.ability_st.a = 0;
                m.ability_boosts = [0; 7];
                Res::Undef
            }
            // onEnd()
            (ab::OPPORTUNIST, Ev::End, Pre::On) => {
                let m = self.mon_mut(holder);
                m.ability_st.a = 0;
                m.ability_boosts = [0; 7];
                Res::Undef
            }

            // ---- Overcoat
            // onImmunity(type, pokemon)
            (ab::OVERCOAT, Ev::Immunity, Pre::On) => {
                if e.imm == Some(Imm::Powder) {
                    return FALSE;
                }
                Res::Undef
            }
            // onTryHit(target, source, move)
            (ab::OVERCOAT, Ev::TryHit, Pre::On) => {
                if mflags & F_POWDER != 0 && e.target != e.source && self.type_allows(holder, 5) {
                    return Res::Null;
                }
                Res::Undef
            }

            // ---- Own Tempo
            // onUpdate(pokemon)
            (ab::OWNTEMPO, Ev::Update, Pre::On) => {
                if self.mon(holder).volatiles.has(VolKind::Confusion) {
                    self.remove_volatile(holder, VolKind::Confusion);
                }
                Res::Undef
            }
            // onTryAddVolatile(status, pokemon)
            (ab::OWNTEMPO, Ev::TryAddVolatile, Pre::On) => {
                if e.vol == Some(VolKind::Confusion) {
                    return Res::Null;
                }
                Res::Undef
            }
            (ab::OWNTEMPO, Ev::Hit, Pre::On) => Res::Undef,

            // ---- Parental Bond
            // onPrepareHit(source, target, move)
            (ab::PARENTALBOND, Ev::PrepareHit, Pre::On) => {
                let Some(m) = mi else {
                    return Res::Undef;
                };
                let am = &mut self.am[m as usize];
                if am.category == Category::Status
                    || am.multihit != (0, 0)
                    || am.flags & (F_NOPARENTALBOND | F_CHARGE | F_FUTUREMOVE) != 0
                    || am.spread_hit
                {
                    return Res::Undef;
                }
                am.multihit = (2, 2);
                am.parental_bond = true;
                Res::Undef
            }
            // onSourceModifySecondaries only concerns Secret Power, which is not in Champions.
            (ab::PARENTALBOND, Ev::ModifySecondaries, Pre::Source) => Res::Undef,

            // ---- Pickpocket: onAfterMoveSecondary(target, source, move)
            (ab::PICKPOCKET, Ev::AfterMoveSecondary, Pre::On) => {
                let (Some(target), Some(source)) = (e.target, e.source) else {
                    return Res::Undef;
                };
                if source == target || mflags & F_CONTACT == 0 {
                    return Res::Undef;
                }
                if self.mon(target).item != it::NONE || self.mon(target).switch_flag || self.mon(source).switch_flag {
                    return Res::Undef;
                }
                let taken = self.take_item(source, Some(target));
                if taken == it::NONE {
                    return Res::Undef;
                }
                if !self.set_item(target, taken, None, Eff::None) {
                    self.mon_mut(source).item = taken;
                }
                Res::Undef
            }

            // ---- Pickup: onResidual(pokemon)
            (ab::PICKUP, Ev::Residual, Pre::On) => {
                if self.mon(holder).item != it::NONE {
                    return Res::Undef;
                }
                let (actives, n) = self.all_active(false);
                let mut cands = [holder; 4];
                let mut k = 0;
                for &t in &actives[..n] {
                    let m = self.mon(t);
                    if m.last_item != it::NONE && m.used_item_this_turn && self.is_adjacent(holder, t) {
                        cands[k] = t;
                        k += 1;
                    }
                }
                if k == 0 {
                    return Res::Undef;
                }
                let from = cands[self.rand(k as u32, "pickup") as usize];
                let item = self.mon(from).last_item;
                self.mon_mut(from).last_item = it::NONE;
                self.set_item(holder, item, None, Eff::None);
                Res::Undef
            }

            // ---- Piercing Drill / Unseen Fist (Champions): onHitProtect(source, target, move).
            // Contact moves go through Protect for a quarter of the damage.
            (ab::PIERCINGDRILL | ab::UNSEENFIST, Ev::HitProtect, Pre::On) => {
                if let (Some(m), Some(target), true) = (mi, e.source, mflags & F_CONTACT != 0) {
                    let slot = self.slot_index(target);
                    self.am[m as usize].hit_data[slot].bypass_protect = true;
                    return FALSE;
                }
                Res::Undef
            }

            // ---- Poison Heal: onDamage(damage, target, source, effect)
            (ab::POISONHEAL, Ev::Damage, Pre::On) => {
                if matches!(e.effect, Eff::Status(Status::Psn) | Eff::Status(Status::Tox)) {
                    let amount = div1(self.mon(holder).max_hp() as u32, 8);
                    self.heal(amount, None, None, Eff::None);
                    return FALSE;
                }
                Res::Undef
            }

            // ---- Poison Touch: onSourceDamagingHit(damage, target, source, move)
            (ab::POISONTOUCH, Ev::DamagingHit, Pre::Source) => {
                let (Some(m), Some(target)) = (mi, e.target) else {
                    return Res::Undef;
                };
                // Shield Dust blocks it even though it is not a secondary effect.
                if self.has_ability(target, ab::SHIELDDUST) {
                    return Res::Undef;
                }
                if self.makes_contact(m) && self.chance(3, 10, "poison touch") {
                    self.try_set_status(target, Status::Psn, e.source, Eff::None);
                }
                Res::Undef
            }

            // ---- Prankster: onModifyPriority(priority, pokemon, target, move)
            (ab::PRANKSTER, Ev::ModifyPriority, Pre::On) => {
                if let (Some(m), Some(Category::Status)) = (mi, mcat) {
                    self.am[m as usize].prankster_boosted = true;
                    return Res::Num(relay.num() + 1);
                }
                Res::Undef
            }

            // ---- Pressure: onDeductPP(target, source)
            (ab::PRESSURE, Ev::DeductPP, Pre::On) => {
                if let (Some(target), Some(source)) = (e.target, e.source) {
                    if self.is_ally(target, source) {
                        return Res::Undef;
                    }
                }
                Res::Num(1)
            }

            // ---- Punk Rock
            // onBasePower(basePower, attacker, defender, move)
            (ab::PUNKROCK, Ev::BasePower, Pre::On) => {
                if mflags & F_SOUND != 0 {
                    return self.chain_modify(5325, 4096);
                }
                Res::Undef
            }
            // onSourceModifyDamage(damage, source, target, move)
            (ab::PUNKROCK, Ev::ModifyDamage, Pre::Source) => {
                if mflags & F_SOUND != 0 {
                    return self.chain_modify(2048, 4096);
                }
                Res::Undef
            }

            // ---- Purifying Salt
            // onSetStatus(status, target, source, effect)
            (ab::PURIFYINGSALT, Ev::SetStatus, Pre::On) => FALSE,
            // onSourceModifyAtk / onSourceModifySpA(atk, attacker, defender, move)
            (ab::PURIFYINGSALT, Ev::ModifyAtk | Ev::ModifySpA, Pre::Source) => {
                if mtype == Some(Type::Ghost) {
                    return self.chain_modify(2048, 4096);
                }
                Res::Undef
            }

            // ---- Quick Draw: onFractionalPriority(priority, pokemon, target, move)
            (ab::QUICKDRAW, Ev::FractionalPriority, Pre::On) => {
                if mcat != Some(Category::Status) && self.chance(3, 10, "quick draw") {
                    return Res::Num(1);
                }
                Res::Undef
            }

            // ---- Quick Feet: onModifySpe(spe, pokemon)
            (ab::QUICKFEET, Ev::ModifySpe, Pre::On) => {
                if self.mon(holder).status != Status::None {
                    return self.chain_modify(6144, 4096);
                }
                Res::Undef
            }

            // ---- Rattled
            // onDamagingHit(damage, target, source, move)
            (ab::RATTLED, Ev::DamagingHit, Pre::On) => {
                if matches!(mtype, Some(Type::Dark | Type::Bug | Type::Ghost)) {
                    self.boost1(SPE, 1, None, None, Eff::None);
                }
                Res::Undef
            }
            // onAfterBoost(boost, target, source, effect)
            (ab::RATTLED, Ev::AfterBoost, Pre::On) => {
                if e.effect == Eff::Ability(ab::INTIMIDATE) && e.boosts[ATK] != 0 {
                    self.boost1(SPE, 1, None, None, Eff::None);
                }
                Res::Undef
            }

            // ---- Receiver: onAllyFaint(target)
            (ab::RECEIVER, Ev::Faint, Pre::Ally) => {
                let Some(fainted) = e.target else {
                    return Res::Undef;
                };
                if self.mon(holder).hp == 0 {
                    return Res::Undef;
                }
                let theirs = self.mon(fainted).ability;
                if ABILITIES[theirs as usize].flags & AF_NORECEIVER != 0 || theirs == ab::NOABILITY {
                    return Res::Undef;
                }
                self.set_ability(holder, theirs, Some(fainted), Eff::None);
                Res::Undef
            }

            // ---- Reckless: onBasePower(basePower, attacker, defender, move)
            (ab::RECKLESS, Ev::BasePower, Pre::On) => {
                if mi.is_some_and(|m| self.am[m as usize].d().recoil.0 > 0) {
                    return self.chain_modify(4915, 4096);
                }
                Res::Undef
            }

            // ---- Regenerator (Champions): onSwitchOut(pokemon)
            (ab::REGENERATOR, Ev::SwitchOut, Pre::On) => {
                let amount = self.mon(holder).max_hp() as i32 / 3;
                self.heal_mon(holder, amount);
                Res::Undef
            }

            // ---- Ripen. `ability_st.a` is `berryWeaken`.
            // onTryHeal(damage, target, source, effect)
            (ab::RIPEN, Ev::TryHeal, Pre::On) => {
                if let Eff::Item(i) = e.effect {
                    if ITEMS[i as usize].flags & IF_BERRY != 0 {
                        return self.chain_modify(2, 1);
                    }
                }
                Res::Undef
            }
            // onChangeBoost(boost, target, source, effect)
            (ab::RIPEN, Ev::ChangeBoost, Pre::On) => {
                if let Eff::Item(i) = e.effect {
                    if ITEMS[i as usize].flags & IF_BERRY != 0 {
                        for b in self.event.boosts.iter_mut() {
                            *b *= 2;
                        }
                    }
                }
                Res::Undef
            }
            // onSourceModifyDamage(damage, source, target, move)
            (ab::RIPEN, Ev::ModifyDamage, Pre::Source) => {
                if self.mon(holder).ability_st.a != 0 {
                    self.mon_mut(holder).ability_st.a = 0;
                    return self.chain_modify(2048, 4096);
                }
                Res::Undef
            }
            (ab::RIPEN, Ev::TryEatItem, Pre::On) => Res::Undef,
            // onEatItem(item, pokemon)
            (ab::RIPEN, Ev::EatItem, Pre::On) => {
                self.mon_mut(holder).ability_st.a = crate::items::is_resist_berry(e.item) as i16;
                Res::Undef
            }

            // ---- Rivalry: onBasePower(basePower, attacker, defender, move)
            (ab::RIVALRY, Ev::BasePower, Pre::On) => {
                let (Some(attacker), Some(defender)) = (e.target, e.source) else {
                    return Res::Undef;
                };
                let (a, d) = (self.mon(attacker).gender, self.mon(defender).gender);
                if a != Gender::N && d != Gender::N {
                    return if a == d { self.chain_modify(5120, 4096) } else { self.chain_modify(3072, 4096) };
                }
                Res::Undef
            }

            // ---- Rock Head: onDamage(damage, target, source, effect)
            (ab::ROCKHEAD, Ev::Damage, Pre::On) => {
                if e.effect == Eff::Recoil {
                    let struggling =
                        self.active_move.is_some_and(|m| self.am[m as usize].d().special == Special::Struggle);
                    if !struggling {
                        return Res::Null;
                    }
                }
                Res::Undef
            }

            // ---- Rough Skin: onDamagingHit(damage, target, source, move)
            (ab::ROUGHSKIN, Ev::DamagingHit, Pre::On) => {
                if let (Some(m), Some(target), Some(source)) = (mi, e.target, e.source) {
                    if self.makes_contact(m) {
                        let d = div1(self.mon(source).max_hp() as u32, 8);
                        self.damage(d, Some(source), Some(target), Eff::None);
                    }
                }
                Res::Undef
            }

            // ---- Run Away (Champions): onTrapPokemon / onMaybeTrapPokemon(pokemon)
            (ab::RUNAWAY, Ev::TrapPokemon, Pre::On) => {
                self.mon_mut(holder).trapped = Trapped::No;
                Res::Undef
            }
            (ab::RUNAWAY, Ev::MaybeTrapPokemon, Pre::On) => Res::Undef,

            // ---- Scrappy: onModifyMove(move)
            (ab::SCRAPPY, Ev::ModifyMove, Pre::On) => {
                if let Some(m) = mi {
                    if self.am[m as usize].ignore_immunity == IgnoreImm::No {
                        self.am[m as usize].ignore_immunity = IgnoreImm::NormalFighting;
                    }
                }
                Res::Undef
            }

            // ---- Serene Grace: onModifyMove(move)
            (ab::SERENEGRACE, Ev::ModifyMove, Pre::On) => {
                if let Some(m) = mi {
                    let am = &mut self.am[m as usize];
                    if am.has_secs {
                        for s in am.secs[..am.n_secs as usize].iter_mut() {
                            s.chance *= 2;
                        }
                    }
                    am.self_chance = am.self_chance.saturating_mul(2);
                }
                Res::Undef
            }

            // ---- Shadow Tag
            // onFoeTrapPokemon(pokemon)
            (ab::SHADOWTAG, Ev::TrapPokemon, Pre::Foe) => {
                if let Some(pokemon) = e.target {
                    if !self.has_ability(pokemon, ab::SHADOWTAG) && self.is_adjacent(pokemon, holder) {
                        self.try_trap(pokemon, true);
                    }
                }
                Res::Undef
            }
            // onFoeMaybeTrapPokemon only decides what the player is told.
            (ab::SHADOWTAG, Ev::MaybeTrapPokemon, Pre::Foe) => Res::Undef,

            // ---- Shed Skin: onResidual(pokemon)
            (ab::SHEDSKIN, Ev::Residual, Pre::On) => {
                let m = self.mon(holder);
                if m.hp > 0 && m.status != Status::None && self.chance(33, 100, "shed skin") {
                    self.cure_status(holder);
                }
                Res::Undef
            }

            // ---- Sheer Force
            // onModifyMove(move, pokemon)
            (ab::SHEERFORCE, Ev::ModifyMove, Pre::On) => {
                if let Some(m) = mi {
                    let am = &mut self.am[m as usize];
                    if am.has_secs {
                        am.has_secs = false;
                        am.n_secs = 0;
                        // Not a secondary effect, but removed with them.
                        am.self_boosts = None;
                        am.self_chance = 0;
                        am.has_sheer_force = true;
                    }
                }
                Res::Undef
            }
            // onBasePower(basePower, pokemon, target, move)
            (ab::SHEERFORCE, Ev::BasePower, Pre::On) => {
                if mi.is_some_and(|m| self.am[m as usize].has_sheer_force) {
                    return self.chain_modify(5325, 4096);
                }
                Res::Undef
            }

            // ---- Shield Dust: onModifySecondaries(secondaries): only those with a `self` part survive.
            (ab::SHIELDDUST, Ev::ModifySecondaries, Pre::On) => {
                if let Some(m) = mi {
                    let am = &self.am[m as usize];
                    for k in 0..am.n_secs as usize {
                        if am.secs[k].self_boosts.is_none() {
                            self.event.secs &= !(1 << k);
                        }
                    }
                }
                Res::Undef
            }

            // ---- Skill Link: onModifyMove(move)
            (ab::SKILLLINK, Ev::ModifyMove, Pre::On) => {
                if let Some(m) = mi {
                    let am = &mut self.am[m as usize];
                    if am.multihit.0 != am.multihit.1 {
                        am.multihit = (am.multihit.1, am.multihit.1);
                    }
                }
                Res::Undef
            }

            // ---- Sniper: onModifyDamage(damage, source, target, move)
            (ab::SNIPER, Ev::ModifyDamage, Pre::On) => {
                if let (Some(m), Some(target)) = (mi, e.source) {
                    if self.am[m as usize].hit_data[self.slot_index(target)].crit {
                        return self.chain_modify(6144, 4096);
                    }
                }
                Res::Undef
            }

            // ---- Soundproof: onTryHit(target, source, move)
            (ab::SOUNDPROOF, Ev::TryHit, Pre::On) => {
                if e.target != e.source && mflags & F_SOUND != 0 {
                    return Res::Null;
                }
                Res::Undef
            }

            // ---- Speed Boost: onResidual(pokemon)
            (ab::SPEEDBOOST, Ev::Residual, Pre::On) => {
                if self.mon(holder).active_turns > 0 {
                    self.boost1(SPE, 1, None, None, Eff::None);
                }
                Res::Undef
            }

            // ---- Spicy Spray: onDamagingHit(damage, target, source, move)
            (ab::SPICYSPRAY, Ev::DamagingHit, Pre::On) => {
                if let Some(source) = e.source {
                    self.try_set_status(source, Status::Brn, e.target, Eff::None);
                }
                Res::Undef
            }

            // ---- Stakeout: onModifyAtk / onModifySpA(atk, attacker, defender)
            (ab::STAKEOUT, Ev::ModifyAtk | Ev::ModifySpA, Pre::On) => {
                if e.source.is_some_and(|d| self.mon(d).active_turns == 0) {
                    return self.chain_modify(2, 1);
                }
                Res::Undef
            }

            // ---- Stall: onFractionalPriority: -0.1
            (ab::STALL, Ev::FractionalPriority, Pre::On) => Res::Num(-1),

            // ---- Stalwart: onModifyMove(move)
            (ab::STALWART, Ev::ModifyMove, Pre::On) => {
                if let Some(m) = mi {
                    self.am[m as usize].tracks_target = self.am[m as usize].target != Target::Scripted;
                }
                Res::Undef
            }

            // ---- Stamina: onDamagingHit(damage, target, source, effect)
            (ab::STAMINA, Ev::DamagingHit, Pre::On) => {
                self.boost1(DEF, 1, None, None, Eff::None);
                Res::Undef
            }

            // ---- Steadfast: onFlinch(pokemon)
            (ab::STEADFAST, Ev::Flinch, Pre::On) => {
                self.boost1(SPE, 1, None, None, Eff::None);
                Res::Undef
            }

            // ---- Steely Spirit: onAllyBasePower(basePower, attacker, defender, move)
            (ab::STEELYSPIRIT, Ev::BasePower, Pre::Ally) => {
                if mtype == Some(Type::Steel) {
                    return self.chain_modify(6144, 4096);
                }
                Res::Undef
            }

            // ---- Stench: onModifyMove(move)
            (ab::STENCH, Ev::ModifyMove, Pre::On) => {
                if let Some(m) = mi {
                    self.add_flinch_secondary(m);
                }
                Res::Undef
            }

            // ---- Sticky Hold: onTakeItem(item, pokemon, source)
            (ab::STICKYHOLD, Ev::TakeItem, Pre::On) => {
                if self.mon(holder).hp == 0 {
                    return Res::Undef;
                }
                if e.source.is_some() && e.source != Some(holder) {
                    return FALSE;
                }
                Res::Undef
            }

            // ---- Sturdy
            // onTryHit only concerns one-hit KO moves, which are not modelled.
            (ab::STURDY, Ev::TryHit, Pre::On) => Res::Undef,
            // onDamage(damage, target, source, effect)
            (ab::STURDY, Ev::Damage, Pre::On) => {
                let m = self.mon(holder);
                if m.hp == m.max_hp() && relay.num() >= m.hp as i32 && e.effect.is_move() {
                    return Res::Num(m.hp as i32 - 1);
                }
                Res::Undef
            }

            // ---- Super Luck: onModifyCritRatio(critRatio)
            (ab::SUPERLUCK, Ev::ModifyCritRatio, Pre::On) => Res::Num(relay.num() + 1),

            // ---- Supersweet Syrup: onStart(pokemon), once per battle.
            (ab::SUPERSWEETSYRUP, Ev::Start, Pre::On) => {
                if self.mon(holder).syrup_triggered {
                    return Res::Undef;
                }
                self.mon_mut(holder).syrup_triggered = true;
                let (foes, n) = self.allies_and_self(1 - holder.side as usize);
                for &f in &foes[..n] {
                    self.boost1(EVA, -1, Some(f), Some(holder), Eff::None);
                }
                Res::Undef
            }

            // ---- Supreme Overlord. `ability_st.a` is the number of fallen allies counted at switch-in.
            // onStart(pokemon)
            (ab::SUPREMEOVERLORD, Ev::Start, Pre::On) => {
                let fallen = self.sides[holder.side as usize].total_fainted.min(5);
                self.mon_mut(holder).ability_st.a = fallen as i16;
                Res::Undef
            }
            (ab::SUPREMEOVERLORD, Ev::End, Pre::On) => Res::Undef,
            // onBasePower(basePower, attacker, defender, move)
            (ab::SUPREMEOVERLORD, Ev::BasePower, Pre::On) => {
                const POW_MOD: [u32; 6] = [4096, 4506, 4915, 5325, 5734, 6144];
                let fallen = self.mon(holder).ability_st.a;
                if fallen > 0 {
                    return self.chain_modify(POW_MOD[fallen as usize], 4096);
                }
                Res::Undef
            }

            // ---- Sweet Veil: onAllySetStatus(status, target, source, effect)
            (ab::SWEETVEIL, Ev::SetStatus, Pre::Ally) => {
                if e.status == Status::Slp {
                    return Res::Null;
                }
                Res::Undef
            }

            // ---- Symbiosis: onAllyAfterUseItem(item, pokemon)
            (ab::SYMBIOSIS, Ev::AfterUseItem, Pre::Ally) => {
                let Some(pokemon) = e.target else {
                    return Res::Undef;
                };
                if self.mon(pokemon).switch_flag {
                    return Res::Undef;
                }
                let mine = self.take_item(holder, None);
                if mine == it::NONE {
                    return Res::Undef;
                }
                if !self.set_item(pokemon, mine, None, Eff::None) {
                    self.mon_mut(holder).item = mine;
                }
                Res::Undef
            }

            // ---- Synchronize: onAfterSetStatus(status, target, source, effect)
            (ab::SYNCHRONIZE, Ev::AfterSetStatus, Pre::On) => {
                let (Some(target), Some(source)) = (e.target, e.source) else {
                    return Res::Undef;
                };
                if source == target || matches!(e.status, Status::Slp | Status::Frz) {
                    return Res::Undef;
                }
                self.try_set_status(source, e.status, Some(target), Eff::Ability(ab::SYNCHRONIZE));
                Res::Undef
            }

            // ---- Tangled Feet: onModifyAccuracy(accuracy, target)
            (ab::TANGLEDFEET, Ev::ModifyAccuracy, Pre::On) => {
                if matches!(relay, Res::Num(_)) && self.mon(holder).volatiles.has(VolKind::Confusion) {
                    return self.chain_modify(2048, 4096);
                }
                Res::Undef
            }

            // ---- Technician: onBasePower(basePower, attacker, defender, move)
            (ab::TECHNICIAN, Ev::BasePower, Pre::On) => {
                if modify(relay.num() as u32, self.event.modifier) <= 60 {
                    return self.chain_modify(6144, 4096);
                }
                Res::Undef
            }

            // ---- Telepathy: onTryHit(target, source, move)
            (ab::TELEPATHY, Ev::TryHit, Pre::On) => {
                if let (Some(target), Some(source)) = (e.target, e.source) {
                    if target != source && self.is_ally(target, source) && mcat != Some(Category::Status) {
                        return Res::Null;
                    }
                }
                Res::Undef
            }

            // ---- Thermal Exchange: onDamagingHit(damage, target, source, move)
            (ab::THERMALEXCHANGE, Ev::DamagingHit, Pre::On) => {
                if mtype == Some(Type::Fire) {
                    self.boost1(ATK, 1, None, None, Eff::None);
                }
                Res::Undef
            }

            // ---- Thick Fat: onSourceModifyAtk / onSourceModifySpA(atk, attacker, defender, move)
            (ab::THICKFAT, Ev::ModifyAtk | Ev::ModifySpA, Pre::Source) => {
                if matches!(mtype, Some(Type::Ice | Type::Fire)) {
                    return self.chain_modify(2048, 4096);
                }
                Res::Undef
            }

            // ---- Trace. `ability_st.a` is `effectState.seek`.
            // onStart(pokemon)
            (ab::TRACE, Ev::Start, Pre::On) => {
                let (foes, n) = self.allies_and_self(1 - holder.side as usize);
                let blank = foes[..n].iter().any(|&f| self.mon(f).ability == ab::NOABILITY);
                self.mon_mut(holder).ability_st.a = !blank as i16;
                if !blank {
                    self.single_event(
                        Ev::Update,
                        Eff::Ability(ab::TRACE),
                        Some(holder),
                        Some(holder),
                        None,
                        Eff::None,
                        Res::Undef,
                    );
                }
                Res::Undef
            }
            // onUpdate(pokemon)
            (ab::TRACE, Ev::Update, Pre::On) => {
                if self.mon(holder).ability_st.a == 0 {
                    return Res::Undef;
                }
                let (foes, n) = self.allies_and_self(1 - holder.side as usize);
                let mut cands = [holder; ACTIVE];
                let mut k = 0;
                for &f in &foes[..n] {
                    let a = self.mon(f).ability;
                    if ABILITIES[a as usize].flags & AF_NOTRACE == 0 && a != ab::NOABILITY {
                        cands[k] = f;
                        k += 1;
                    }
                }
                if k == 0 {
                    return Res::Undef;
                }
                let target = cands[self.rand(k as u32, "trace") as usize];
                let ability = self.mon(target).ability;
                self.set_ability(holder, ability, Some(target), Eff::None);
                Res::Undef
            }

            // ---- Unaware: onAnyModifyBoost(boosts, pokemon)
            (ab::UNAWARE, Ev::ModifyBoost, Pre::Any) => {
                let pokemon = e.target;
                if pokemon == Some(holder) {
                    return Res::Undef;
                }
                if Some(holder) == self.active_pokemon && pokemon == self.active_target {
                    self.event.boosts[DEF] = 0;
                    self.event.boosts[SPD] = 0;
                    self.event.boosts[EVA] = 0;
                }
                if pokemon == self.active_pokemon && Some(holder) == self.active_target {
                    self.event.boosts[ATK] = 0;
                    self.event.boosts[DEF] = 0;
                    self.event.boosts[SPA] = 0;
                    self.event.boosts[ACC] = 0;
                }
                Res::Undef
            }

            // ---- Unburden
            // onAfterUseItem(item, pokemon)
            (ab::UNBURDEN, Ev::AfterUseItem, Pre::On) => {
                if e.target == Some(holder) {
                    self.add_volatile(holder, VolKind::Unburden, None, Eff::None);
                }
                Res::Undef
            }
            // onTakeItem(item, pokemon)
            (ab::UNBURDEN, Ev::TakeItem, Pre::On) => {
                self.add_volatile(holder, VolKind::Unburden, None, Eff::None);
                Res::Undef
            }
            // onEnd(pokemon)
            (ab::UNBURDEN, Ev::End, Pre::On) => {
                self.remove_volatile(holder, VolKind::Unburden);
                Res::Undef
            }

            // ---- Unnerve. `ability_st.a` is `effectState.unnerved`.
            (ab::UNNERVE, Ev::Start, Pre::On) => {
                self.mon_mut(holder).ability_st.a = 1;
                Res::Undef
            }
            (ab::UNNERVE, Ev::End, Pre::On) => {
                self.mon_mut(holder).ability_st.a = 0;
                Res::Undef
            }
            // onFoeTryEatItem()
            (ab::UNNERVE, Ev::TryEatItem, Pre::Foe) => Res::Bool(self.mon(holder).ability_st.a == 0),

            // ---- Wandering Spirit: onDamagingHit(damage, target, source, move)
            (ab::WANDERINGSPIRIT, Ev::DamagingHit, Pre::On) => {
                if let (Some(m), Some(target), Some(source)) = (mi, e.target, e.source) {
                    if self.makes_contact(m) {
                        self.skill_swap(source, target);
                    }
                }
                Res::Undef
            }

            // ---- Water Bubble
            // onSourceModifyAtk / onSourceModifySpA(atk, attacker, defender, move)
            (ab::WATERBUBBLE, Ev::ModifyAtk | Ev::ModifySpA, Pre::Source) => {
                if mtype == Some(Type::Fire) {
                    return self.chain_modify(2048, 4096);
                }
                Res::Undef
            }
            // onModifyAtk / onModifySpA(atk, attacker, defender, move)
            (ab::WATERBUBBLE, Ev::ModifyAtk | Ev::ModifySpA, Pre::On) => {
                if mtype == Some(Type::Water) {
                    return self.chain_modify(2, 1);
                }
                Res::Undef
            }

            // ---- Weak Armor: onDamagingHit(damage, target, source, move)
            (ab::WEAKARMOR, Ev::DamagingHit, Pre::On) => {
                if mcat == Some(Category::Physical) {
                    let mut b = [0i8; 7];
                    b[DEF] = -1;
                    b[SPE] = 2;
                    self.boost(b, e.target, e.target, Eff::None);
                }
                Res::Undef
            }

            _ => unreachable!("no body for ability {} {ev:?} {pre:?}", ABILITIES[ability as usize].id),
        }
    }
}
