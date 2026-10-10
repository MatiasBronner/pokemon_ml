//! Event callbacks of held items, ported from Showdown's `data/items.ts` with
//! the Champions overrides.
//!
//! Each arm is one Showdown callback; the comment gives its name. Inside a
//! callback `self.event` holds the event's target, source, causing effect and
//! relay value, exactly as Showdown passes them as arguments, and `holder` is
//! the Pokémon holding the item (`this.effectState.target`).

#![allow(clippy::collapsible_if)]

use crate::battle::div1;
use crate::data::*;
use crate::state::*;

/// The type a type-boosting item (Charcoal, Mystic Water, ...) strengthens.
fn type_booster(item: u16) -> Option<Type> {
    Some(match item {
        it::BLACKBELT => Type::Fighting,
        it::BLACKGLASSES => Type::Dark,
        it::CHARCOAL => Type::Fire,
        it::DRAGONFANG => Type::Dragon,
        it::FAIRYFEATHER => Type::Fairy,
        it::HARDSTONE => Type::Rock,
        it::MAGNET => Type::Electric,
        it::METALCOAT => Type::Steel,
        it::MIRACLESEED => Type::Grass,
        it::MYSTICWATER => Type::Water,
        it::NEVERMELTICE => Type::Ice,
        it::POISONBARB => Type::Poison,
        it::SHARPBEAK => Type::Flying,
        it::SILKSCARF => Type::Normal,
        it::SILVERPOWDER => Type::Bug,
        it::SOFTSAND => Type::Ground,
        it::SPELLTAG => Type::Ghost,
        it::TWISTEDSPOON => Type::Psychic,
        _ => return None,
    })
}

/// The type a damage-halving berry (Occa, Passho, ...) weakens. Chilan Berry
/// (Normal) is the only one that does not need the hit to be super effective.
fn resist_berry(item: u16) -> Option<Type> {
    Some(match item {
        it::BABIRIBERRY => Type::Steel,
        it::CHARTIBERRY => Type::Rock,
        it::CHILANBERRY => Type::Normal,
        it::CHOPLEBERRY => Type::Fighting,
        it::COBABERRY => Type::Flying,
        it::COLBURBERRY => Type::Dark,
        it::HABANBERRY => Type::Dragon,
        it::KASIBBERRY => Type::Ghost,
        it::KEBIABERRY => Type::Poison,
        it::OCCABERRY => Type::Fire,
        it::PASSHOBERRY => Type::Water,
        it::PAYAPABERRY => Type::Psychic,
        it::RINDOBERRY => Type::Grass,
        it::ROSELIBERRY => Type::Fairy,
        it::SHUCABERRY => Type::Ground,
        it::TANGABERRY => Type::Bug,
        it::WACANBERRY => Type::Electric,
        it::YACHEBERRY => Type::Ice,
        _ => return None,
    })
}

pub(crate) fn is_resist_berry(item: u16) -> bool {
    resist_berry(item).is_some()
}

/// The status a status-curing berry removes.
fn cure_berry(item: u16) -> Option<&'static [Status]> {
    Some(match item {
        it::ASPEARBERRY => &[Status::Frz],
        it::CHERIBERRY => &[Status::Par],
        it::CHESTOBERRY => &[Status::Slp],
        it::PECHABERRY => &[Status::Psn, Status::Tox],
        it::RAWSTBERRY => &[Status::Brn],
        _ => return None,
    })
}

/// The volatiles Mental Herb cures. None of them is modelled yet; looking them
/// up by name means the herb starts working the moment one is.
pub(crate) const MENTAL_HERB: [&str; 6] = ["attract", "taunt", "encore", "torment", "disable", "healblock"];

impl Battle {
    /// The move the running event is about, if it is about one.
    pub(crate) fn event_move(&self) -> Option<u8> {
        match self.event.effect {
            Eff::Move(mi) => Some(mi),
            _ => None,
        }
    }

    /// Runs one item callback.
    pub(crate) fn item_cb(&mut self, item: u16, ev: Ev, pre: Pre, holder: MonRef) -> Res {
        let e = self.event;
        let relay = e.relay;

        // Families of items that share one callback.
        if !ITEMS[item as usize].mega.is_empty() {
            // Mega Stones: onTakeItem(item, source). The Pokémon Showdown passes
            // second is the holder; the stone cannot be taken from a Pokémon
            // whose base species it belongs to.
            debug_assert!(ev == Ev::TakeItem && pre == Pre::On);
            let d = &ITEMS[item as usize];
            // (When a move asks the stone directly, "the holder" is whoever would end up with it.)
            let species = self.mon(e.target.unwrap_or(holder)).base_species;
            let own = if d.flags & IF_MEGA_BY_FORME != 0 {
                // A few stones go by the exact forme instead: the ones they evolve and the Megas they make.
                d.mega.iter().any(|&(from, to)| from == species || to == species)
            } else {
                let base = SPECIES[species as usize].base_species;
                d.mega.iter().any(|&(from, _)| SPECIES[from as usize].id == base)
            };
            return Res::Bool(!own);
        }
        if let Some(t) = type_booster(item) {
            // onBasePower(basePower, user, target, move)
            debug_assert!(ev == Ev::BasePower && pre == Pre::On);
            if self.event_move().is_some_and(|mi| self.am[mi as usize].typ == t) {
                return self.chain_modify(4915, 4096);
            }
            return Res::Undef;
        }
        if let Some(t) = resist_berry(item) {
            return match ev {
                // onSourceModifyDamage(damage, source, target, move): the holder is the move's target.
                Ev::ModifyDamage => {
                    let Some(mi) = self.event_move() else {
                        return Res::Undef;
                    };
                    let Some(target) = e.source else {
                        return Res::Undef;
                    };
                    let slot = self.slot_index(target);
                    let am = &self.am[mi as usize];
                    if am.typ == t && (item == it::CHILANBERRY || am.hit_data[slot].type_mod > 0) {
                        // Not for a hit its substitute is about to take.
                        let hit_sub = self.vols(target).has(VolKind::Substitute)
                            && am.flags & F_BYPASSSUB == 0
                            && !am.infiltrates;
                        if hit_sub {
                            return Res::Undef;
                        }
                        if self.eat_item(target, false, None, Eff::None) {
                            return self.chain_modify(2048, 4096);
                        }
                    }
                    Res::Undef
                }
                // onEat() {}
                Ev::Eat => Res::Undef,
                _ => unreachable!("no body for item {} {ev:?}", ITEMS[item as usize].id),
            };
        }
        if let Some(cures) = cure_berry(item) {
            return match ev {
                // onUpdate(pokemon)
                Ev::Update => {
                    if cures.contains(&self.mon(holder).status) {
                        self.eat_item(holder, false, None, Eff::None);
                    }
                    Res::Undef
                }
                // onEat(pokemon)
                Ev::Eat => {
                    if cures.contains(&self.mon(holder).status) {
                        self.cure_status(holder);
                    }
                    Res::Undef
                }
                _ => unreachable!("no body for item {} {ev:?}", ITEMS[item as usize].id),
            };
        }

        // The terrain seeds: used up, for a stat boost, while their terrain is up.
        let seed = match item {
            it::ELECTRICSEED => Some(Terrain::Electricterrain),
            it::GRASSYSEED => Some(Terrain::Grassyterrain),
            it::MISTYSEED => Some(Terrain::Mistyterrain),
            it::PSYCHICSEED => Some(Terrain::Psychicterrain),
            _ => None,
        };
        if let Some(terrain) = seed {
            return match ev {
                // onStart(pokemon)
                Ev::Start => {
                    if !self.ignoring_item(holder) && self.is_terrain(terrain) {
                        self.use_item(holder, None, Eff::None);
                    }
                    Res::Undef
                }
                // onTerrainChange(pokemon)
                Ev::TerrainChange => {
                    if self.is_terrain(terrain) {
                        self.use_item(holder, None, Eff::None);
                    }
                    Res::Undef
                }
                _ => unreachable!("no body for item {} {ev:?}", ITEMS[item as usize].id),
            };
        }
        match (item, ev, pre) {
            // ---- Air Balloon (the levitation itself is in `is_grounded`)
            // onStart(target): `-item|target|Air Balloon`, the one item that announces itself.
            (it::AIRBALLOON, Ev::Start, Pre::On) => {
                if !self.ignoring_item(holder) && !self.field.pseudo.has(Pseudo::Gravity) {
                    self.show_item_held(holder, item);
                }
                Res::Undef
            }
            // onDamagingHit: popped by any damaging hit, without counting as "used".
            // onAfterSubDamage(damage, target, source, effect): and by a move that hits its substitute.
            (it::AIRBALLOON, Ev::DamagingHit | Ev::AfterSubDamage, Pre::On) => {
                let Some(target) = e.target else {
                    return Res::Undef;
                };
                if !matches!(e.effect, Eff::Move(_)) {
                    return Res::Undef;
                }
                // `-enditem|target|Air Balloon`
                self.show_item_lost(target, item);
                {
                    let m = self.mon_mut(target);
                    m.item = it::NONE;
                    m.item_st.order = 0;
                    m.item_st.a = 0;
                    m.item_st.b = 0;
                }
                let mut ae = Event::new(Ev::AfterUseItem, Some(target), None, Eff::None);
                ae.item = it::AIRBALLOON;
                self.run_event_ex(ae, Res::Undef, false, false);
                Res::Undef
            }

            // ---- Big Root: onTryHeal(damage, target, source, effect)
            (it::BIGROOT, Ev::TryHeal, Pre::On) => {
                // Draining moves, Leech Seed, Ingrain, Aqua Ring and Strength Sap.
                if matches!(e.effect, Eff::Drain | Eff::Vol(VolKind::Leechseed | VolKind::Ingrain | VolKind::Aquaring))
                    || self.eff_is_named(e.effect, "strengthsap")
                {
                    return self.chain_modify(5324, 4096);
                }
                Res::Undef
            }

            // ---- Bright Powder: onModifyAccuracy(accuracy)
            (it::BRIGHTPOWDER, Ev::ModifyAccuracy, Pre::On) => {
                if matches!(relay, Res::Num(_)) {
                    return self.chain_modify(3686, 4096);
                }
                Res::Undef
            }

            // ---- Choice Scarf
            // onStart(pokemon)
            (it::CHOICESCARF, Ev::Start, Pre::On) => {
                self.remove_volatile(holder, VolKind::Choicelock);
                Res::Undef
            }
            // onModifyMove(move, pokemon)
            (it::CHOICESCARF, Ev::ModifyMove, Pre::On) => {
                self.add_volatile(holder, VolKind::Choicelock, None, Eff::None);
                Res::Undef
            }
            // onModifySpe(spe, pokemon)
            (it::CHOICESCARF, Ev::ModifySpe, Pre::On) => self.chain_modify(6144, 4096),

            // ---- Expert Belt: onModifyDamage(damage, source, target, move)
            (it::EXPERTBELT, Ev::ModifyDamage, Pre::On) => {
                if let (Some(mi), Some(target)) = (self.event_move(), e.source) {
                    let slot = self.slot_index(target);
                    if self.am[mi as usize].hit_data[slot].type_mod > 0 {
                        return self.chain_modify(4915, 4096);
                    }
                }
                Res::Undef
            }

            // ---- Focus Band: onDamage(damage, target, source, effect). The
            // 10% roll comes first, so it is made for every bit of damage.
            (it::FOCUSBAND, Ev::Damage, Pre::On) => {
                let Some(target) = e.target else {
                    return Res::Undef;
                };
                if self.chance(1, 10, "focus band") && relay.num() >= self.mon(target).hp as i32 && e.effect.is_move() {
                    // `-activate|target|item: Focus Band`
                    self.show_item(target, item);
                    return Res::Num(self.mon(target).hp as i32 - 1);
                }
                Res::Undef
            }
            // ---- Focus Sash: onDamage(damage, target, source, effect)
            (it::FOCUSSASH, Ev::Damage, Pre::On) => {
                let Some(target) = e.target else {
                    return Res::Undef;
                };
                let m = self.mon(target);
                if m.hp == m.max_hp() && relay.num() >= m.hp as i32 && e.effect.is_move() {
                    if self.use_item(target, None, Eff::None) {
                        return Res::Num(self.mon(target).hp as i32 - 1);
                    }
                }
                Res::Undef
            }

            // ---- Iron Ball (grounding is in `is_grounded`)
            // onEffectiveness(typeMod, target, type, move)
            (it::IRONBALL, Ev::Effectiveness, Pre::On) => {
                let (Some(target), Some(mi)) = (e.target, self.event_move()) else {
                    return Res::Undef;
                };
                // (When something else already grounds the holder, that rule decides.)
                if self.has_vol_named(target, "ingrain")
                    || self.has_vol_named(target, "smackdown")
                    || self.field.pseudo.has(Pseudo::Gravity)
                {
                    return Res::Undef;
                }
                if self.am[mi as usize].typ == Type::Ground && self.has_type(target, Type::Flying) {
                    return Res::Num(0);
                }
                Res::Undef
            }
            // onModifySpe(spe)
            (it::IRONBALL, Ev::ModifySpe, Pre::On) => self.chain_modify(2048, 4096),

            // ---- King's Rock: onModifyMove(move)
            (it::KINGSROCK, Ev::ModifyMove, Pre::On) => {
                if let Some(mi) = self.event_move() {
                    self.add_flinch_secondary(mi);
                }
                Res::Undef
            }

            // ---- Leek: onModifyCritRatio(critRatio, user)
            (it::LEEK, Ev::ModifyCritRatio, Pre::On) => {
                let base = e.target.map_or("", |u| SPECIES[self.mon(u).base_species as usize].base_species);
                if base == "farfetchd" || base == "sirfetchd" {
                    return Res::Num(relay.num() + 2);
                }
                Res::Undef
            }

            // ---- Leftovers: onResidual(pokemon)
            (it::LEFTOVERS, Ev::Residual, Pre::On) => {
                let amount = div1(self.mon(holder).max_hp() as u32, 16);
                self.heal(amount, None, None, Eff::None);
                Res::Undef
            }

            // ---- Leppa Berry
            // onUpdate(pokemon)
            (it::LEPPABERRY, Ev::Update, Pre::On) => {
                let m = self.mon(holder);
                if m.hp == 0 {
                    return Res::Undef;
                }
                if m.moves[..m.n_moves as usize].iter().any(|s| s.pp == 0) {
                    self.eat_item(holder, false, None, Eff::None);
                }
                Res::Undef
            }
            // onEat(pokemon)
            (it::LEPPABERRY, Ev::Eat, Pre::On) => {
                let added = if self.has_ability(holder, ab::RIPEN) { 20 } else { 10 };
                let m = self.mon_mut(holder);
                let n = m.n_moves as usize;
                let slot = m.moves[..n]
                    .iter()
                    .position(|s| s.pp == 0)
                    .or_else(|| m.moves[..n].iter().position(|s| s.pp < s.maxpp));
                if let Some(k) = slot {
                    m.moves[k].pp = (m.moves[k].pp + added).min(m.moves[k].maxpp);
                    // `-activate|pokemon|item: Leppa Berry|Move`
                    let restored = m.moves[k].id;
                    self.show_move(holder, restored);
                }
                Res::Undef
            }

            // ---- Eject Button: onAfterMoveSecondary(target, source, move). The holder leaves when hit.
            (it::EJECTBUTTON, Ev::AfterMoveSecondary, Pre::On) => {
                let (Some(target), Some(source), Some(mi)) = (e.target, e.source, self.event_move()) else {
                    return Res::Undef;
                };
                let am = &self.am[mi as usize];
                if source == target
                    || self.mon(target).hp == 0
                    || am.category == Category::Status
                    || am.flags & F_FUTUREMOVE != 0
                {
                    return Res::Undef;
                }
                if !self.can_switch(target.side as usize) || self.mon(target).force_switch_flag {
                    return Res::Undef;
                }
                // Only one such exit at a time.
                let (actives, n) = self.all_active(false);
                if actives[..n].iter().any(|&p| self.switch_flag_is_true(p)) {
                    return Res::Undef;
                }
                self.mon_mut(target).switch_flag = true;
                self.mon_mut(target).switch_move = NO_MOVE;
                if !self.use_item(target, None, Eff::None) {
                    self.mon_mut(target).switch_flag = false;
                }
                Res::Undef
            }

            // ---- Red Card: onAfterMoveSecondary(target, source, move). The attacker is dragged out.
            (it::REDCARD, Ev::AfterMoveSecondary, Pre::On) => {
                let (Some(target), Some(source), Some(mi)) = (e.target, e.source, self.event_move()) else {
                    return Res::Undef;
                };
                if source == target
                    || self.mon(source).hp == 0
                    || self.mon(target).hp == 0
                    || self.am[mi as usize].category == Category::Status
                {
                    return Res::Undef;
                }
                if !self.mon(source).is_active
                    || !self.can_switch(source.side as usize)
                    || self.mon(source).force_switch_flag
                    || self.mon(target).force_switch_flag
                {
                    return Res::Undef;
                }
                if self.use_item(target, Some(source), Eff::None)
                    && self.run_event(Ev::DragOut, Some(source), Some(target), Eff::Move(mi), Res::Undef).truthy()
                {
                    self.mon_mut(source).force_switch_flag = true;
                }
                Res::Undef
            }

            // ---- Life Orb
            // onModifyDamage(damage, source, target, move)
            (it::LIFEORB, Ev::ModifyDamage, Pre::On) => self.chain_modify(5324, 4096),
            // onAfterMoveSecondarySelf(source, target, move)
            (it::LIFEORB, Ev::AfterMoveSecondarySelf, Pre::On) => {
                let (Some(source), Some(mi)) = (e.target, self.event_move()) else {
                    return Res::Undef;
                };
                if Some(source) != e.source
                    && self.am[mi as usize].category != Category::Status
                    && !self.mon(source).force_switch_flag
                {
                    let d = div1(self.mon(source).max_hp() as u32, 10);
                    self.damage(d, Some(source), Some(source), Eff::Item(it::LIFEORB));
                }
                Res::Undef
            }

            // ---- Light Ball: onModifyAtk / onModifySpA (atk, pokemon)
            (it::LIGHTBALL, Ev::ModifyAtk | Ev::ModifySpA, Pre::On) => {
                let base = e.target.map_or("", |u| SPECIES[self.mon(u).base_species as usize].base_species);
                if base == "pikachu" {
                    return self.chain_modify(2, 1);
                }
                Res::Undef
            }

            // ---- Lum Berry
            // onAfterSetStatus(status, pokemon)
            (it::LUMBERRY, Ev::AfterSetStatus, Pre::On) => {
                self.eat_item(holder, false, None, Eff::None);
                Res::Undef
            }
            // onUpdate(pokemon)
            (it::LUMBERRY, Ev::Update, Pre::On) => {
                let m = self.mon(holder);
                if m.status != Status::None || self.vols(holder).has(VolKind::Confusion) {
                    self.eat_item(holder, false, None, Eff::None);
                }
                Res::Undef
            }
            // onEat(pokemon)
            (it::LUMBERRY, Ev::Eat, Pre::On) => {
                self.cure_status(holder);
                self.remove_volatile(holder, VolKind::Confusion);
                Res::Undef
            }

            // ---- Mental Herb: onUpdate(pokemon)
            (it::MENTALHERB, Ev::Update, Pre::On) => {
                let afflicted = |b: &Battle, name: &str| b.vols(holder).as_slice().iter().any(|v| v.kind.id() == name);
                if MENTAL_HERB.iter().any(|name| afflicted(self, name)) {
                    if !self.use_item(holder, None, Eff::None) {
                        return Res::Undef;
                    }
                    for name in MENTAL_HERB {
                        let kind = self.vols(holder).as_slice().iter().map(|v| v.kind).find(|k| k.id() == name);
                        if let Some(k) = kind {
                            self.remove_volatile(holder, k);
                        }
                    }
                }
                Res::Undef
            }

            // ---- Metronome: onStart(pokemon)
            (it::METRONOME, Ev::Start, Pre::On) => {
                self.add_volatile(holder, VolKind::Metronome, None, Eff::None);
                Res::Undef
            }

            // ---- Muscle Band / Wise Glasses: onBasePower(basePower, user, target, move)
            (it::MUSCLEBAND, Ev::BasePower, Pre::On) => {
                if self.event_move().is_some_and(|mi| self.am[mi as usize].category == Category::Physical) {
                    return self.chain_modify(4505, 4096);
                }
                Res::Undef
            }
            (it::WISEGLASSES, Ev::BasePower, Pre::On) => {
                if self.event_move().is_some_and(|mi| self.am[mi as usize].category == Category::Special) {
                    return self.chain_modify(4505, 4096);
                }
                Res::Undef
            }

            // ---- Normal Gem: onSourceTryPrimaryHit(target, source, move)
            (it::NORMALGEM, Ev::TryPrimaryHit, Pre::Source) => {
                let (Some(target), Some(source), Some(mi)) = (e.target, e.source, self.event_move()) else {
                    return Res::Undef;
                };
                let am = &self.am[mi as usize];
                if target == source || am.category == Category::Status {
                    return Res::Undef;
                }
                if am.typ == Type::Normal && self.use_item(source, None, Eff::None) {
                    self.add_volatile(source, VolKind::Gem, None, Eff::None);
                }
                Res::Undef
            }

            // ---- Oran Berry / Sitrus Berry
            // onUpdate(pokemon)
            (it::ORANBERRY | it::SITRUSBERRY, Ev::Update, Pre::On) => {
                let m = self.mon(holder);
                if m.hp as u32 * 2 <= m.max_hp() as u32 {
                    self.eat_item(holder, false, None, Eff::None);
                }
                Res::Undef
            }
            // onTryEatItem(item, pokemon): not eaten if the healing would be blocked.
            (it::ORANBERRY | it::SITRUSBERRY, Ev::TryEatItem, Pre::On) => {
                let amount = if item == it::ORANBERRY { 10 } else { div1(self.mon(holder).max_hp() as u32, 4) };
                if !self.run_event(Ev::TryHeal, Some(holder), None, Eff::Item(item), Res::Num(amount)).truthy() {
                    return FALSE;
                }
                Res::Undef
            }
            // onEat(pokemon)
            (it::ORANBERRY | it::SITRUSBERRY, Ev::Eat, Pre::On) => {
                let amount = if item == it::ORANBERRY { 10 } else { div1(self.mon(holder).max_hp() as u32, 4) };
                self.heal(amount, None, None, Eff::None);
                Res::Undef
            }

            // ---- Persim Berry
            (it::PERSIMBERRY, Ev::Update, Pre::On) => {
                if self.vols(holder).has(VolKind::Confusion) {
                    self.eat_item(holder, false, None, Eff::None);
                }
                Res::Undef
            }
            (it::PERSIMBERRY, Ev::Eat, Pre::On) => {
                self.remove_volatile(holder, VolKind::Confusion);
                Res::Undef
            }

            // ---- Quick Claw: onFractionalPriority(priority, pokemon, target, move)
            (it::QUICKCLAW, Ev::FractionalPriority, Pre::On) => {
                if relay.num() <= 0 && self.chance(1, 5, "quick claw") {
                    // `-activate|pokemon|item: Quick Claw`
                    self.show_item(holder, item);
                    return Res::Num(1);
                }
                Res::Undef
            }

            // ---- Rocky Helmet: onDamagingHit(damage, target, source, move)
            (it::ROCKYHELMET, Ev::DamagingHit, Pre::On) => {
                let (Some(target), Some(source), Some(mi)) = (e.target, e.source, self.event_move()) else {
                    return Res::Undef;
                };
                if self.makes_contact(mi) {
                    let d = div1(self.mon(source).max_hp() as u32, 6);
                    self.damage(d, Some(source), Some(target), Eff::None);
                }
                Res::Undef
            }

            // ---- Scope Lens: onModifyCritRatio(critRatio)
            (it::SCOPELENS, Ev::ModifyCritRatio, Pre::On) => Res::Num(relay.num() + 1),

            // ---- Shed Shell: onTrapPokemon / onMaybeTrapPokemon (pokemon)
            (it::SHEDSHELL, Ev::TrapPokemon, Pre::On) => {
                self.mon_mut(holder).trapped = Trapped::No;
                Res::Undef
            }
            (it::SHEDSHELL, Ev::MaybeTrapPokemon, Pre::On) => Res::Undef,

            // ---- Shell Bell: onAfterMoveSecondarySelf(pokemon, target, move)
            (it::SHELLBELL, Ev::AfterMoveSecondarySelf, Pre::On) => {
                let (Some(pokemon), Some(mi)) = (e.target, self.event_move()) else {
                    return Res::Undef;
                };
                let total = self.am[mi as usize].total_damage;
                if total > 0 && !self.mon(pokemon).force_switch_flag {
                    self.heal(div1(total as u32, 8), Some(pokemon), None, Eff::None);
                }
                Res::Undef
            }

            // ---- White Herb
            // onStart and the callbacks that simply re-run it.
            (it::WHITEHERB, Ev::Start, Pre::On)
            | (it::WHITEHERB, Ev::SwitchIn, Pre::Any)
            | (it::WHITEHERB, Ev::AfterMega, Pre::Any)
            | (it::WHITEHERB, Ev::AfterMove, Pre::Any)
            | (it::WHITEHERB, Ev::Residual, Pre::On) => {
                if self.mon(holder).boosts.iter().any(|&b| b < 0) {
                    self.use_item(holder, None, Eff::None);
                }
                Res::Undef
            }
            // onUse(pokemon): reset the lowered stats.
            (it::WHITEHERB, Ev::Use, Pre::On) => {
                for b in self.mon_mut(holder).boosts.iter_mut() {
                    if *b < 0 {
                        *b = 0;
                    }
                }
                Res::Undef
            }

            // ---- Wide Lens / Zoom Lens: onSourceModifyAccuracy(accuracy, target)
            (it::WIDELENS, Ev::ModifyAccuracy, Pre::Source) => {
                if matches!(relay, Res::Num(_)) {
                    return self.chain_modify(4505, 4096);
                }
                Res::Undef
            }
            (it::ZOOMLENS, Ev::ModifyAccuracy, Pre::Source) => {
                if matches!(relay, Res::Num(_)) && e.target.is_some_and(|t| !self.will_move(t)) {
                    return self.chain_modify(4915, 4096);
                }
                Res::Undef
            }

            _ => unreachable!("no body for item {} {ev:?} {pre:?}", ITEMS[item as usize].id),
        }
    }

    /// King's Rock and Stench: give a damaging move a 10% flinch chance unless
    /// it can already make the target flinch.
    pub(crate) fn add_flinch_secondary(&mut self, mi: u8) {
        let am = &mut self.am[mi as usize];
        if am.category == Category::Status {
            return;
        }
        let n = am.n_secs as usize;
        if !am.has_secs {
            am.has_secs = true;
        }
        if am.secs[..n].iter().any(|s| s.volatile == Some(VolKind::Flinch)) {
            return;
        }
        if n < MAX_SECS {
            am.secs[n] = Secondary {
                chance: 10,
                status: Status::None,
                boosts: None,
                volatile: Some(VolKind::Flinch),
                self_boosts: None,
                on_hit: false,
            };
            am.n_secs += 1;
        }
    }
}
