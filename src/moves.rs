//! Using a move: a port of Showdown's `sim/battle-actions.ts` with the
//! Champions overrides (`spreadMoveHit`, `hitStepMoveHitLoop`, `modifyDamage`).

#![allow(clippy::needless_range_loop, clippy::collapsible_if, clippy::collapsible_match, clippy::if_same_then_else)]

use crate::battle::*;
use crate::data::*;
use crate::state::*;

/// What one application of a move does to its target: the move itself, one of
/// its secondaries, or a `self` block (Showdown's `HitEffect`).
#[derive(Clone, Copy)]
pub(crate) struct HitEff {
    /// The move itself rather than a secondary or self effect.
    primary: bool,
    boosts: Option<Boosts>,
    boost_order: u8,
    status: Status,
    volatile: Option<VolKind>,
    heal: (u8, u8),
    self_boosts: Option<Boosts>,
    self_chance: u8,
}

impl HitEff {
    fn of_move(am: &ActiveMove) -> HitEff {
        let d = am.d();
        HitEff {
            primary: true,
            boosts: d.boosts,
            boost_order: d.boost_order,
            status: d.status,
            volatile: d.volatile,
            heal: d.heal,
            self_boosts: am.self_boosts,
            self_chance: am.self_chance,
        }
    }
    fn of_secondary(s: &Secondary) -> HitEff {
        HitEff {
            primary: false,
            boosts: s.boosts,
            boost_order: 0,
            status: s.status,
            volatile: s.volatile,
            heal: (0, 0),
            self_boosts: s.self_boosts,
            self_chance: 0,
        }
    }
    fn of_self(b: Boosts) -> HitEff {
        HitEff {
            primary: false,
            boosts: Some(b),
            boost_order: 0,
            status: Status::None,
            volatile: None,
            heal: (0, 0),
            self_boosts: None,
            self_chance: 0,
        }
    }
}

impl Battle {
    /// Whether a move has a script callback of its own for `ev`.
    pub(crate) fn move_has_cb(&self, mi: u8, ev: Ev) -> bool {
        match self.am[mi as usize].d().special {
            Special::None => false,
            Special::Protect => matches!(ev, Ev::PrepareHit | Ev::Hit),
            Special::Struggle => ev == Ev::ModifyMove,
        }
    }

    /// The script callbacks of moves the engine implements by hand.
    pub(crate) fn move_cb(&mut self, mi: u8, ev: Ev) -> Res {
        let target = self.event.target;
        match (self.am[mi as usize].d().special, ev) {
            // protect.onPrepareHit
            (Special::Protect, Ev::PrepareHit) => {
                if !self.will_act() {
                    return FALSE;
                }
                self.run_event(Ev::StallMove, target, None, Eff::None, Res::Undef)
            }
            // protect.onHit
            (Special::Protect, Ev::Hit) => {
                if let Some(t) = target {
                    self.add_volatile(t, VolKind::Stall, None, Eff::None);
                }
                Res::Undef
            }
            // struggle.onModifyMove
            (Special::Struggle, Ev::ModifyMove) => {
                self.am[mi as usize].typ = Type::Typeless;
                Res::Undef
            }
            _ => Res::Undef,
        }
    }

    /// `Pokemon#deductPP`: returns how much PP was actually removed.
    pub(crate) fn deduct_pp(&mut self, r: MonRef, move_id: u16, amount: u32) -> u32 {
        let m = self.mon_mut(r);
        let n = m.n_moves as usize;
        let Some(s) = m.moves[..n].iter_mut().find(|s| s.id == move_id) else {
            return 0;
        };
        if s.pp == 0 {
            return 0;
        }
        let taken = amount.min(s.pp as u32);
        s.pp -= taken as u8;
        taken
    }

    /// `Battle#suppressingSecondaries`.
    fn suppressing_secondaries(&self) -> bool {
        match (self.active_move, self.active_pokemon) {
            (Some(mi), Some(p)) => self.am[mi as usize].has_sheer_force && self.has_ability(p, ab::SHEERFORCE),
            _ => false,
        }
    }

    /// `Battle#checkMoveMakesContact`.
    pub(crate) fn makes_contact(&self, mi: u8) -> bool {
        self.am[mi as usize].flags & F_CONTACT != 0
    }

    /// `Battle#checkMoveBypassesProtect`.
    pub(crate) fn bypasses_protect(&mut self, mi: u8, attacker: MonRef, defender: MonRef, block_status: bool) -> bool {
        let am = &self.am[mi as usize];
        if (am.category != Category::Status || block_status)
            && am.flags & F_PROTECT != 0
            && self.run_event(Ev::HitProtect, Some(attacker), Some(defender), Eff::Move(mi), Res::Undef).truthy()
        {
            return false;
        }
        true
    }

    /// `BattleActions#runMove`: the "outside" of using a move (can it move at
    /// all, PP), for a move chosen as this turn's action.
    pub(crate) fn run_move(&mut self, a: &Action) {
        let pokemon = a.mon.unwrap();
        let d = &MOVES[a.move_id as usize];
        let target = self.get_target(pokemon, d.target, a.target_loc, a.orig_target);
        let mi = self.new_am(a.move_id);
        self.am[mi as usize].priority = a.move_priority;
        self.am[mi as usize].prankster_boosted = a.prankster;
        // OverrideAction: nothing modelled listens.
        self.set_active_move(Some(mi), Some(pokemon), target);
        let will_try = self.run_event(Ev::BeforeMove, Some(pokemon), target, Eff::Move(mi), Res::Undef);
        if !will_try.truthy() {
            self.run_event(Ev::MoveAborted, Some(pokemon), target, Eff::Move(mi), Res::Undef);
            self.clear_active_move(true);
            return;
        }
        if self.deduct_pp(pokemon, a.move_id, 1) == 0 && d.special != Special::Struggle {
            self.clear_active_move(true);
            return;
        }
        self.use_move(mi, pokemon, target, Eff::None);
        let mv = self.active_move.unwrap_or(mi);
        self.run_event(Ev::AfterMove, Some(pokemon), target, Eff::Move(mv), Res::Undef);
        self.faint_messages(false, false, true);
        self.check_win(None);
    }

    /// `BattleActions#useMove` / `useMoveInner`: the effects of the move itself.
    pub(crate) fn use_move(&mut self, mi: u8, pokemon: MonRef, target: Option<MonRef>, source_effect: Eff) -> bool {
        let m = mi as usize;
        let mut target = target;
        let mut source_effect = source_effect;
        if source_effect == Eff::None && self.effect != Eff::None {
            source_effect = self.effect;
        }
        if let Some(cur) = self.active_move {
            if cur != mi {
                self.am[m].priority = self.am[cur as usize].priority;
                if !self.am[m].has_bounced {
                    self.am[m].prankster_boosted = self.am[cur as usize].prankster_boosted;
                }
            }
        }
        let base_target = self.am[m].target;
        // ModifyTarget: nothing modelled listens.
        if matches!(self.am[m].target, Target::User | Target::Allies) {
            target = Some(pokemon);
        }
        if source_effect != Eff::None {
            self.am[m].source_effect = source_effect;
            self.am[m].ignore_ability = match source_effect {
                Eff::Move(s) => self.am[s as usize].ignore_ability,
                _ => false,
            };
        }
        self.set_active_move(Some(mi), Some(pokemon), target);

        let me = Eff::Move(mi);
        self.single_event(Ev::ModifyType, me, None, Some(pokemon), target, me, Res::Undef);
        self.single_event(Ev::ModifyMove, me, None, Some(pokemon), target, me, Res::Undef);
        if base_target != self.am[m].target {
            target = self.get_random_target(pokemon, self.am[m].target);
        }
        self.run_event(Ev::ModifyType, Some(pokemon), target, me, Res::Undef);
        self.run_event(Ev::ModifyMove, Some(pokemon), target, me, Res::Undef);
        if base_target != self.am[m].target {
            target = self.get_random_target(pokemon, self.am[m].target);
        }
        if self.mon(pokemon).fainted {
            return false;
        }
        let Some(chosen) = target else {
            return false;
        };
        let (targets, n) = self.get_move_targets(pokemon, mi, chosen);
        if n > 0 {
            target = Some(targets[n - 1]);
        }
        if source_effect == Eff::None && self.listens(Ev::DeductPP) {
            // Pressure
            let mut extra = 0;
            for &t in &targets[..n] {
                let drop = self.run_event(Ev::DeductPP, Some(t), Some(pokemon), me, Res::Undef);
                if drop != TRUE {
                    extra += drop.num();
                }
            }
            if extra > 0 {
                let id = self.am[m].id;
                self.deduct_pp(pokemon, id, extra as u32);
            }
        }

        let mut try_move = self.single_event(Ev::TryMove, me, None, Some(pokemon), target, me, Res::Undef);
        if try_move.truthy() {
            try_move = self.run_event(Ev::TryMove, Some(pokemon), target, me, Res::Undef);
        }
        if !try_move.truthy() {
            return false;
        }

        if n == 0 {
            return false;
        }
        let result = self.try_spread_move_hit(&targets[..n], pokemon, mi);
        if self.mon(pokemon).hp == 0 {
            self.faint(pokemon, Some(pokemon), me);
        }
        if !result {
            return false;
        }
        if !self.suppressing_secondaries() {
            self.run_event(Ev::AfterMoveSecondarySelf, Some(pokemon), target, me, Res::Undef);
        }
        true
    }

    /// `Pokemon#getMoveTargets`.
    fn get_move_targets(&mut self, user: MonRef, mi: u8, target: MonRef) -> ([MonRef; MAX_TARGETS], usize) {
        let mut out = [user; MAX_TARGETS];
        let mut n = 0;
        let own = user.side as usize;
        let move_target = self.am[mi as usize].target;
        match move_target {
            Target::AllAdjacent | Target::AllAdjacentFoes => {
                if move_target == Target::AllAdjacent {
                    let (allies, k) = self.adjacent_allies(user);
                    for &a in &allies[..k] {
                        out[n] = a;
                        n += 1;
                    }
                }
                let (foes, k) = self.allies_and_self(1 - own);
                for &f in &foes[..k] {
                    out[n] = f;
                    n += 1;
                }
            }
            Target::Allies => {
                let (allies, k) = self.allies_and_self(own);
                for &a in &allies[..k] {
                    out[n] = a;
                    n += 1;
                }
            }
            _ => {
                let mut t = target;
                if self.mon(t).fainted && t.side != user.side {
                    // A foe that fainted earlier this turn: retarget.
                    match self.get_random_target(user, move_target) {
                        Some(p) => t = p,
                        None => return (out, 0),
                    }
                }
                if !self.am[mi as usize].tracks_target && self.listens(Ev::RedirectTarget) {
                    let r = self.priority_event(Ev::RedirectTarget, Some(user), Some(user), Eff::Move(mi), Res::Mon(t));
                    if let Res::Mon(redirected) = r {
                        t = redirected;
                    }
                }
                if self.mon(t).fainted {
                    return (out, 0);
                }
                out[0] = t;
                n = 1;
            }
        }
        (out, n)
    }

    /// Showdown filters `targets` after each hit step by "truthy or 0".
    fn keep_hits(targets: &mut [MonRef; MAX_TARGETS], n: &mut usize, res: &Damage, any_failure: &mut bool) {
        let mut w = 0;
        for i in 0..*n {
            if res[i] == FALSE {
                *any_failure = true;
            }
            if res[i].hit() {
                targets[w] = targets[i];
                w += 1;
            }
        }
        *n = w;
    }

    /// `BattleActions#trySpreadMoveHit` (this is also the path single-target moves take).
    fn try_spread_move_hit(&mut self, initial: &[MonRef], user: MonRef, mi: u8) -> bool {
        let m = mi as usize;
        let me = Eff::Move(mi);
        let mut targets = [user; MAX_TARGETS];
        let mut n = initial.len();
        targets[..n].copy_from_slice(initial);
        if n > 1 {
            self.am[m].spread_hit = true;
        }

        // `Try` has no listeners among modelled moves.
        let mut hit_result = self.single_event(Ev::PrepareHit, me, None, Some(targets[0]), Some(user), me, Res::Undef);
        if hit_result.truthy() {
            hit_result = self.run_event(Ev::PrepareHit, Some(user), Some(targets[0]), me, Res::Undef);
        }
        if !hit_result.truthy() {
            return hit_result == Res::NotFail;
        }

        let mut failure = false;
        let mut res: Damage = [TRUE; MAX_TARGETS];

        // Step 0: semi-invulnerability.
        let toxic_sure = self.am[m].d().id == "toxic" && self.has_type(user, Type::Poison);
        for i in 0..n {
            res[i] = if toxic_sure {
                TRUE
            } else {
                self.run_event(Ev::Invulnerability, Some(targets[i]), Some(user), me, Res::Undef)
            };
        }
        Battle::keep_hits(&mut targets, &mut n, &res, &mut failure);
        if n == 0 {
            return false;
        }

        // Step 1: the TryHit event (Protect, absorbing abilities, ...).
        self.run_event_multi(Ev::TryHit, &targets[..n], Some(user), me, &mut res[..n], false);
        for i in 0..n {
            if res[i] != Res::NotFail && !res[i].truthy() {
                res[i] = FALSE;
            }
        }
        Battle::keep_hits(&mut targets, &mut n, &res, &mut failure);
        if n == 0 {
            return false;
        }

        // Step 2: type immunity.
        for i in 0..n {
            res[i] = Res::Bool(self.run_immunity(targets[i], mi));
        }
        Battle::keep_hits(&mut targets, &mut n, &res, &mut failure);
        if n == 0 {
            return false;
        }

        // Step 3: move-specific immunities.
        for i in 0..n {
            let t = targets[i];
            let am = &self.am[m];
            let powder = am.flags & F_POWDER != 0;
            let prankster = am.prankster_boosted;
            res[i] = if powder && t != user && !self.type_allows(t, 5) {
                FALSE
            } else if prankster
                && self.has_ability(user, ab::PRANKSTER)
                && !self.is_ally(t, user)
                && !self.type_allows(t, 7)
            {
                FALSE
            } else {
                TRUE
            };
        }
        Battle::keep_hits(&mut targets, &mut n, &res, &mut failure);
        if n == 0 {
            return false;
        }

        // Step 4: accuracy.
        for i in 0..n {
            res[i] = Res::Bool(self.accuracy_check(targets[i], user, mi));
        }
        Battle::keep_hits(&mut targets, &mut n, &res, &mut failure);
        if n == 0 {
            return false;
        }

        // Steps 5 and 6 (breaking protection, stealing boosts): no modelled move does either.

        // Step 7: the hits themselves.
        let dmg = self.move_hit_loop(&targets[..n], user, mi);
        Battle::keep_hits(&mut targets, &mut n, &dmg, &mut failure);
        n > 0
    }

    /// One target's part of `hitStepAccuracy`: does the move hit it?
    fn accuracy_check(&mut self, target: MonRef, user: MonRef, mi: u8) -> bool {
        let m = mi as usize;
        let me = Eff::Move(mi);
        self.active_target = Some(target);
        let base = self.am[m].accuracy;
        let mut accuracy = if base == 0 { TRUE } else { Res::Num(base as i32) };
        accuracy = self.run_event(Ev::ModifyAccuracy, Some(target), Some(user), me, accuracy);
        if let Res::Num(a) = accuracy {
            let ub = self.mon(user).boosts;
            let mut boost = (self.modify_boost(user, ub)[ACC] as i32).clamp(-6, 6);
            if !self.am[m].ignore_evasion {
                let tb = self.mon(target).boosts;
                boost = (boost - self.modify_boost(target, tb)[EVA] as i32).clamp(-6, 6);
            }
            let mut a = a;
            if boost > 0 {
                a = a * (3 + boost) / 3;
            } else if boost < 0 {
                a = a * 3 / (3 - boost);
            }
            accuracy = Res::Num(a);
        }
        let am = &self.am[m];
        let toxic_sure = am.d().id == "toxic" && self.has_type(user, Type::Poison);
        if toxic_sure || (am.target == Target::User && am.category == Category::Status) {
            accuracy = TRUE;
        } else {
            accuracy = self.run_event(Ev::Accuracy, Some(target), Some(user), me, accuracy);
        }
        match accuracy {
            Res::Bool(true) => true,
            Res::Num(a) => self.chance(a.max(0) as u32, 100, "accuracy"),
            _ => self.chance(0, 100, "accuracy"),
        }
    }

    /// `hitStepMoveHitLoop` as overridden by the Champions mod.
    fn move_hit_loop(&mut self, targets: &[MonRef], user: MonRef, mi: u8) -> Damage {
        let m = mi as usize;
        let n = targets.len();
        let mut damage: Damage = [Res::Num(0); MAX_TARGETS];
        self.am[m].total_damage = 0;
        let target_hits: u32 = match self.am[m].multihit {
            (0, 0) => 1,
            (a, b) if a == b => a as u32,
            (2, 5) => {
                const HITS: [u32; 20] = [2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5];
                HITS[self.rand(20, "multihit count") as usize]
            }
            (a, b) => self.rand_range(a as usize, b as usize + 1, "multihit count") as u32,
        };
        let mut copy: Targets = [Tgt::Gone; MAX_TARGETS];
        let mut hit = 1;
        let mut completed_one = false;
        while hit <= target_hits {
            if damage[..n].contains(&FALSE) {
                break;
            }
            if hit > 1 && self.mon(user).status == Status::Slp {
                break;
            }
            if targets.iter().all(|&t| self.mon(t).hp == 0) {
                break;
            }
            self.am[m].hit = hit as u8;
            self.am[m].last_hit = hit == target_hits;
            for i in 0..n {
                copy[i] = Tgt::Mon(targets[i]);
            }
            let eff = HitEff::of_move(&self.am[m]);
            let move_damage = self.spread_move_hit(&mut copy, n, user, mi, eff, false, false);
            if !move_damage[..n].iter().any(|&v| v != FALSE) {
                break;
            }
            for i in 0..n {
                let dealt = move_damage[i].num();
                damage[i] = Res::Num(dealt);
                self.am[m].total_damage += dealt;
            }
            self.each_event(Ev::Update);
            completed_one = true;
            if self.mon(user).hp == 0 && n == 1 {
                break;
            }
            hit += 1;
        }
        if !completed_one {
            return [FALSE; MAX_TARGETS];
        }
        let user_down = self.mon(user).hp == 0;
        self.faint_messages(false, false, user_down);

        if self.am[m].total_damage > 0 {
            self.apply_recoil(self.am[m].total_damage as u32, mi, user);
        }
        if !damage[..n].iter().any(|v| v.hit()) {
            return damage;
        }
        self.each_event(Ev::Update);

        // afterMoveSecondaryEvent on the targets still standing in `copy`.
        let mut left = [user; MAX_TARGETS];
        let mut k = 0;
        for i in 0..n {
            if let Tgt::Mon(t) = copy[i] {
                left[k] = t;
                k += 1;
            }
        }
        if k > 0 {
            let mut relays = [Res::Undef; MAX_TARGETS];
            self.run_event_multi(
                Ev::AfterMoveSecondary,
                &left[..k],
                Some(user),
                Eff::Move(mi),
                &mut relays[..k],
                false,
            );
        }
        damage
    }

    /// `BattleActions#applyRecoilDamage`.
    fn apply_recoil(&mut self, dealt: u32, mi: u8, user: MonRef) {
        let d = self.am[mi as usize].d();
        if d.special == Special::Struggle {
            let r = round_div(self.mon(user).max_hp() as u32, 1, 4).max(1);
            self.direct_damage(r as i32, user, Some(user), Eff::StruggleRecoil);
        } else if d.recoil.0 > 0 {
            let r = round_div(dealt, d.recoil.0 as u32, d.recoil.1 as u32).max(1);
            self.damage(r as i32, Some(user), Some(user), Eff::Recoil);
        }
    }

    /// `BattleActions#spreadMoveHit` as overridden by the Champions mod.
    #[allow(clippy::too_many_arguments)]
    fn spread_move_hit(
        &mut self,
        targets: &mut Targets,
        n: usize,
        user: MonRef,
        mi: u8,
        eff: HitEff,
        is_secondary: bool,
        is_self: bool,
    ) -> Damage {
        let me = Eff::Move(mi);
        let mut damage: Damage = [TRUE; MAX_TARGETS];

        // 0. the TryPrimaryHit event (Substitute, gems)
        if !is_secondary && !is_self && self.listens(Ev::TryPrimaryHit) {
            for i in 0..n {
                if let Tgt::Mon(t) = targets[i] {
                    damage[i] = self.run_event(Ev::TryPrimaryHit, Some(t), Some(user), me, Res::Undef);
                }
            }
        }
        for i in 0..n {
            if targets[i] != Tgt::Gone && is_secondary && eff.self_boosts.is_none() {
                damage[i] = TRUE;
            }
            if !damage[i].truthy() {
                targets[i] = Tgt::Gone;
            }
        }

        // 1. damage calculation
        for i in 0..n {
            if let Tgt::Mon(t) = targets[i] {
                self.active_target = Some(t);
                let cur = self.get_damage(user, t, mi, &eff);
                damage[i] = if cur == FALSE || cur == Res::Null { FALSE } else { cur };
            }
        }
        for i in 0..n {
            if damage[i] == FALSE {
                targets[i] = Tgt::Gone;
            }
        }

        // 2. apply damage
        self.spread_damage(&mut damage, targets, n, Some(user), me);
        for i in 0..n {
            if damage[i] == FALSE {
                targets[i] = Tgt::Gone;
            }
        }

        // 3. the effect's own boosts, healing, status, volatiles and Hit events
        self.run_move_effects(&mut damage, targets, n, user, mi, &eff, is_secondary, is_self);
        for i in 0..n {
            if !damage[i].hit() {
                targets[i] = Tgt::Gone;
            }
        }

        let active_target = self.active_target;

        // 4. stat changes to the user
        if let Some(b) = eff.self_boosts {
            if !self.am[mi as usize].self_dropped {
                self.self_drops(targets, n, user, mi, b, eff.self_chance, eff.primary, is_secondary);
            }
        }

        // 5. secondary effects
        if eff.primary && self.am[mi as usize].has_secs {
            self.secondaries(targets, n, user, mi, is_self);
        }

        self.active_target = active_target;

        for i in 0..n {
            if !damage[i].hit() {
                targets[i] = Tgt::Gone;
            }
        }

        if !is_secondary && !is_self {
            let mut hit = [user; MAX_TARGETS];
            let mut amounts = [Res::Undef; MAX_TARGETS];
            let mut k = 0;
            for i in 0..n {
                if let (Res::Num(_), Tgt::Mon(t)) = (damage[i], targets[i]) {
                    hit[k] = t;
                    amounts[k] = damage[i];
                    k += 1;
                }
            }
            if k > 0 {
                self.run_event_multi(Ev::DamagingHit, &hit[..k], Some(user), me, &mut amounts[..k], true);
            }
        }
        damage
    }

    /// `BattleActions#selfDrops`.
    #[allow(clippy::too_many_arguments)]
    fn self_drops(
        &mut self,
        targets: &Targets,
        n: usize,
        user: MonRef,
        mi: u8,
        b: Boosts,
        chance: u8,
        primary: bool,
        is_secondary: bool,
    ) {
        for i in 0..n {
            if targets[i] == Tgt::Gone || self.am[mi as usize].self_dropped {
                continue;
            }
            let mut apply = true;
            if !is_secondary {
                // Showdown rolls here even when there is no chance to check.
                let roll = self.rand(100, "self stat change");
                apply = !primary || chance == 0 || roll < chance as u32;
                if self.am[mi as usize].multihit == (0, 0) {
                    self.am[mi as usize].self_dropped = true;
                }
            }
            if apply {
                let mut one: Targets = [Tgt::Mon(user), Tgt::Gone, Tgt::Gone];
                self.spread_move_hit(&mut one, 1, user, mi, HitEff::of_self(b), is_secondary, true);
            }
        }
    }

    /// `BattleActions#secondaries`.
    fn secondaries(&mut self, targets: &Targets, n: usize, user: MonRef, mi: u8, is_self: bool) {
        for i in 0..n {
            let Tgt::Mon(t) = targets[i] else {
                continue;
            };
            let count = self.am[mi as usize].n_secs as usize;
            let mut keep: u8 = (1u8 << count) - 1;
            if self.listens(Ev::ModifySecondaries) {
                let mut e = Event::new(Ev::ModifySecondaries, Some(t), Some(user), Eff::Move(mi));
                e.secs = keep;
                keep = self.run_event_ex(e, Res::Undef, false, false).1.secs;
            }
            for k in 0..count {
                if keep & (1 << k) == 0 {
                    continue;
                }
                let sec = self.am[mi as usize].secs[k];
                // The roll happens even when the secondary is guaranteed.
                let roll = self.rand(100, "secondary effect");
                if sec.chance == 0 || roll < sec.chance as u32 {
                    let mut one: Targets = [Tgt::Mon(t), Tgt::Gone, Tgt::Gone];
                    self.spread_move_hit(&mut one, 1, user, mi, HitEff::of_secondary(&sec), true, is_self);
                }
            }
        }
    }

    /// `BattleActions#runMoveEffects`.
    #[allow(clippy::too_many_arguments)]
    fn run_move_effects(
        &mut self,
        damage: &mut Damage,
        targets: &Targets,
        n: usize,
        source: MonRef,
        mi: u8,
        eff: &HitEff,
        is_secondary: bool,
        is_self: bool,
    ) {
        let me = Eff::Move(mi);
        for i in 0..n {
            let Tgt::Mon(t) = targets[i] else {
                continue;
            };
            let mut did = Res::Undef;
            if let Some(b) = eff.boosts {
                if !self.mon(t).fainted {
                    let r = self.boost(b, eff.boost_order, Some(t), Some(source), me);
                    did = did.combine(r);
                }
            }
            if eff.heal.0 > 0 && !self.mon(t).fainted {
                let mon = self.mon(t);
                if mon.hp >= mon.max_hp() {
                    damage[i] = damage[i].combine(FALSE);
                    continue;
                }
                let amount = round_div(mon.max_hp() as u32, eff.heal.0 as u32, eff.heal.1 as u32);
                let healed = self.heal(amount as i32, Some(t), Some(source), me);
                if !healed.hit() {
                    damage[i] = damage[i].combine(FALSE);
                    continue;
                }
                did = TRUE;
            }
            if eff.status != Status::None {
                let r = self.try_set_status(t, eff.status, Some(source), me);
                if !r.truthy() && self.am[mi as usize].d().status != Status::None {
                    damage[i] = damage[i].combine(FALSE);
                    continue;
                }
                did = did.combine(r);
            }
            if let Some(v) = eff.volatile {
                let r = self.add_volatile(t, v, Some(source), me);
                did = did.combine(r);
            }
            if eff.primary && self.move_has_cb(mi, Ev::Hit) {
                let r = self.single_event(Ev::Hit, me, None, Some(t), Some(source), me, Res::Undef);
                did = did.combine(r);
            }
            if !is_self && !is_secondary {
                self.run_event(Ev::Hit, Some(t), Some(source), me, Res::Undef);
            }
            if did == Res::Undef {
                did = TRUE;
            }
            damage[i] = damage[i].combine(if did == Res::Null { FALSE } else { did });
        }
    }

    /// `BattleActions#getDamage` followed by the Champions `modifyDamage`.
    fn get_damage(&mut self, user: MonRef, target: MonRef, mi: u8, eff: &HitEff) -> Res {
        if !eff.primary {
            // Secondaries and self effects carry no type and no base power.
            return Res::Undef;
        }
        let m = mi as usize;
        let me = Eff::Move(mi);
        if !self.run_immunity(target, mi) {
            return FALSE;
        }
        let base_power = self.am[m].base_power as i32;
        if base_power == 0 {
            return Res::Undef;
        }

        const CRIT_MULT: [u32; 5] = [0, 24, 8, 2, 1];
        let crit_ratio = self
            .run_event(Ev::ModifyCritRatio, Some(user), Some(target), me, Res::Num(self.am[m].crit_ratio as i32))
            .num();
        let crit_ratio = crit_ratio.clamp(0, 4) as usize;
        let slot = self.slot_index(target);
        let mut crit = self.am[m].will_crit.unwrap_or(false);
        if self.am[m].will_crit.is_none() && crit_ratio > 0 {
            crit = self.chance(1, CRIT_MULT[crit_ratio], "critical hit");
        }
        if crit {
            crit = self.run_event(Ev::CriticalHit, Some(target), None, me, Res::Undef).truthy();
        }
        self.am[m].hit_data[slot].crit = crit;

        let e = Event::new(Ev::BasePower, Some(user), Some(target), me);
        let base_power = self.run_event_ex(e, Res::Num(base_power), true, false).0.num();
        if base_power == 0 {
            return Res::Num(0);
        }
        let base_power = base_power.max(1) as u32;

        let d = self.am[m].d();
        let physical = self.am[m].category == Category::Physical;
        let atk_stat = if d.off_stat != 0 {
            d.off_stat as usize
        } else if physical {
            1
        } else {
            3
        };
        let def_stat = if d.def_stat != 0 {
            d.def_stat as usize
        } else if physical {
            2
        } else {
            4
        };
        let attacker = if d.off_from_target { target } else { user };
        let mut atk_stage = self.mon(attacker).boosts[atk_stat - 1] as i32;
        let mut def_stage = self.mon(target).boosts[def_stat - 1] as i32;
        // A critical hit ignores the attacker's drops and the defender's boosts.
        if crit && atk_stage < 0 {
            atk_stage = 0;
        }
        if self.am[m].ignore_defensive || (crit && def_stage > 0) {
            def_stage = 0;
        }
        let attack = self.calculate_stat(attacker, atk_stat, atk_stage, user);
        let defense = self.calculate_stat(target, def_stat, def_stage, target);
        // The offensive event is named after the move's category, the
        // defensive one after the stat actually used.
        let atk_ev = if physical { Ev::ModifyAtk } else { Ev::ModifySpA };
        const DEF_EVS: [Ev; 5] = [Ev::ModifyAtk, Ev::ModifyDef, Ev::ModifySpA, Ev::ModifySpD, Ev::ModifySpe];
        let attack = self.run_event(atk_ev, Some(user), Some(target), me, Res::Num(attack as i32)).num() as u32;
        let defense =
            self.run_event(DEF_EVS[def_stat - 1], Some(target), Some(user), me, Res::Num(defense as i32)).num() as u32;
        let level = self.mon(user).level as u32;
        let dmg = (2 * level / 5 + 2).wrapping_mul(base_power).wrapping_mul(attack) / defense.max(1) / 50;
        self.modify_damage(dmg, user, target, mi)
    }

    /// The Champions `modifyDamage`.
    fn modify_damage(&mut self, base: u32, user: MonRef, target: MonRef, mi: u8) -> Res {
        let m = mi as usize;
        let me = Eff::Move(mi);
        let typ = self.am[m].typ;
        let slot = self.slot_index(target);
        let mut dmg = base + 2;
        if self.am[m].spread_hit {
            dmg = modify(dmg, 3072);
        }
        // WeatherModifyDamage: no weather yet.
        let crit = self.am[m].hit_data[slot].crit;
        if crit {
            dmg = dmg * 3 / 2;
        }
        let roll = self.rand(16, "damage roll");
        dmg = dmg.wrapping_mul(100 - roll) / 100;
        if typ != Type::Typeless {
            let mut stab = 4096;
            if self.has_type(user, typ) {
                stab = 6144;
            }
            stab = self.run_event(Ev::ModifySTAB, Some(user), Some(target), me, Res::Num(stab)).num();
            dmg = modify(dmg, stab as u32);
        }
        let type_mod = self.run_effectiveness(target, mi).clamp(-6, 6);
        self.am[m].hit_data[slot].type_mod = type_mod as i8;
        if type_mod > 0 {
            dmg <<= type_mod;
        } else {
            for _ in 0..-type_mod {
                dmg /= 2;
            }
        }
        if self.mon(user).status == Status::Brn
            && self.am[m].category == Category::Physical
            && !self.has_ability(user, ab::GUTS)
            && self.am[m].d().id != "facade"
        {
            dmg = modify(dmg, 2048);
        }
        dmg = self.run_event(Ev::ModifyDamage, Some(user), Some(target), me, Res::Num(dmg as i32)).num() as u32;
        if self.am[m].hit_data[slot].bypass_protect {
            dmg = modify(dmg, 1024);
        }
        if dmg == 0 {
            return Res::Num(1);
        }
        Res::Num((dmg & 0xFFFF) as i32)
    }
}
