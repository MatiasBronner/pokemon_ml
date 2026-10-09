//! The public entry points: building a battle, listing legal choices, and
//! submitting them (Showdown's `Side#chooseMove` / `chooseSwitch` validation
//! and `Battle#commitChoices`).

#![allow(clippy::needless_range_loop, clippy::collapsible_if)]

use crate::battle::*;
use crate::data::*;
use crate::rng::Rng;
use crate::state::*;

impl Battle {
    /// Builds a battle from the four Pokémon each side picked at team preview,
    /// in the order picked (the first two lead), and plays the opening
    /// switch-ins. `seed` is Showdown's four 16-bit seed words.
    pub fn new(teams: [&[PokemonSet]; 2], seed: [u16; 4]) -> Result<Battle, Error> {
        let blank_slot = MoveSlot { id: 0, pp: 0, maxpp: 0, disabled: false, hidden: false, used: false };
        let blank_mon = Pokemon {
            species: 0,
            base_species: 0,
            set_species: 0,
            transformed: false,
            base_moves: [MoveSlot { id: 0, pp: 0, maxpp: 0, disabled: false, hidden: false, used: false }; MAX_MOVES],
            base_n_moves: 0,
            illusion: 0,
            can_mega: NO_SPECIES,
            nature: (0, 0),
            stat_points: [0; 6],
            types: [Type::None; 2],
            added_type: Type::None,
            level: 50,
            gender: Gender::N,
            stats: [0; 6],
            hp: 0,
            status: Status::None,
            status_time: 0,
            tox_stage: 0,
            status_st: EffState::default(),
            boosts: [0; 7],
            moves: [blank_slot; MAX_MOVES],
            n_moves: 0,
            ability: ab::NOABILITY,
            base_ability: ab::NOABILITY,
            ability_st: EffState::default(),
            ability_boosts: [0; 7],
            syrup_triggered: false,
            was_attacked: false,
            last_attack_damage: 0,
            item: it::NONE,
            item_st: EffState::default(),
            last_item: it::NONE,
            used_item_this_turn: false,
            ate_berry: false,
            position: 0,
            is_active: false,
            fainted: true,
            faint_queued: false,
            switch_flag: false,
            switch_move: NO_MOVE,
            force_switch_flag: false,
            skip_before_switch_out: false,
            being_called_back: false,
            trapped: Trapped::No,
            active_turns: 0,
            move_this_turn: Res::Undef,
            move_last_turn: Res::Undef,
            last_move: NO_MOVE,
            last_move_loc: 0,
            active_move_actions: 0,
            newly_switched: true,
            hurt_this_turn: 0,
            times_attacked: 0,
            stats_raised_this_turn: false,
            stats_lowered_this_turn: false,
            hit_by_this_turn: 0,
            damaged_by: [NO_DAMAGED_BY; MAX_TEAM],
            n_damaged_by: 0,
            locked_move: NO_MOVE,
            speed: 0,
        };
        let blank_side = Side {
            team: [blank_mon; MAX_TEAM],
            n: 0,
            order: [0, 1, 2, 3, 4, 5],
            pokemon_left: 0,
            total_fainted: 0,
            fainted_this_turn: false,
            fainted_last_turn: false,
            conds: SideConds::new(SideCond::FIRST),
            slot_conds: [SlotConds::new(SlotCond::FIRST); ACTIVE],
        };
        let nobody = MonRef { side: 0, idx: 0 };
        let mut b = Battle {
            rng: Rng::from_words(seed),
            sides: [blank_side; 2],
            field: Field {
                weather: Cond::new(Weather::None),
                terrain: Cond::new(Terrain::None),
                pseudo: PseudoWeathers::new(Pseudo::FIRST),
            },
            vols: [[Volatiles::new(VolKind::FIRST); ACTIVE]; 2],
            turn: 0,
            request: Request::None,
            ended: false,
            winner: None,
            queue: Queue::new(),
            faint_queue: [FaintEntry { target: nobody, source: None, effect: Eff::None }; 8],
            n_faint: 0,
            mid_turn: true,
            effect_order: 0,
            next_uid: 0,
            event_mask: 0,
            event_mask_pre: 0,
            event: Event::NONE,
            effect: Eff::None,
            effect_holder: None,
            event_depth: 0,
            am: [ActiveMove::new(0); AM_CAP],
            am_len: 0,
            active_move: None,
            last_move: NO_MOVE,
            started: false,
            active_pokemon: None,
            active_target: None,
            speed_order: [0; 4],
            n_speed_order: 0,
        };
        for (s, team) in teams.iter().enumerate() {
            if team.len() < ACTIVE || team.len() > MAX_TEAM {
                return Err(Error::BadTeam(format!(
                    "side {} brings {} Pokémon; need {ACTIVE} to {MAX_TEAM}",
                    s + 1,
                    team.len()
                )));
            }
            for (i, set) in team.iter().enumerate() {
                let sp = SPECIES.get(set.species as usize).ok_or_else(|| Error::BadTeam("unknown species".into()))?;
                if sp.illegal {
                    return Err(Error::Unsupported(format!("{} is not in Champions", sp.name)));
                }
                if set.moves.is_empty() || set.moves.len() > MAX_MOVES {
                    return Err(Error::BadTeam(format!("{} needs 1 to {MAX_MOVES} moves", sp.name)));
                }
                if set.stat_points.iter().any(|&p| p > 32)
                    || set.stat_points.iter().map(|&p| p as u32).sum::<u32>() > 66
                {
                    return Err(Error::BadTeam(format!(
                        "{}: at most 32 stat points per stat and 66 in total",
                        sp.name
                    )));
                }
                let ability =
                    ABILITIES.get(set.ability as usize).ok_or_else(|| Error::BadTeam("unknown ability".into()))?;
                if !ability.supported {
                    return Err(Error::Unsupported(format!("ability {}", ability.name)));
                }
                let item = ITEMS.get(set.item as usize).ok_or_else(|| Error::BadTeam("unknown item".into()))?;
                if !item.supported {
                    return Err(Error::Unsupported(format!("item {}", item.name)));
                }
                b.event_mask |= ability.events | ability.events_pre | item.events | item.events_pre;
                b.event_mask_pre |= ability.events_pre | item.events_pre;
                let mut mon = blank_mon;
                // `canMegaEvo`: holding the Mega Stone of this very species.
                if let Some(&(_, mega)) = item.mega.iter().find(|&&(from, _)| from == set.species) {
                    let mega_ability = &ABILITIES[SPECIES[mega as usize].ability0 as usize];
                    if !mega_ability.supported {
                        return Err(Error::Unsupported(format!(
                            "{} (its ability, {}, is not modelled)",
                            SPECIES[mega as usize].name, mega_ability.name
                        )));
                    }
                    b.event_mask |= mega_ability.events | mega_ability.events_pre;
                    b.event_mask_pre |= mega_ability.events_pre;
                    mon.can_mega = mega;
                }
                mon.species = set.species;
                mon.base_species = set.species;
                mon.set_species = set.species;
                mon.nature = set.nature;
                mon.stat_points = set.stat_points;
                mon.types = sp.types;
                mon.added_type = Type::None;
                mon.gender = set.gender;
                mon.ability = set.ability;
                mon.base_ability = set.ability;
                mon.item = set.item;
                mon.fainted = false;
                mon.position = i as u8;
                // Champions stats: base + stat points + 75 for HP, + 20 otherwise.
                mon.stats = calc_stats(set.species, set.nature, set.stat_points);
                mon.hp = mon.stats[0];
                mon.speed = mon.stats[5] as i32;
                for (k, &id) in set.moves.iter().enumerate() {
                    let d = MOVES.get(id as usize).ok_or_else(|| Error::BadTeam("unknown move".into()))?;
                    if !d.supported {
                        return Err(Error::Unsupported(format!("move {}", d.name)));
                    }
                    mon.moves[k] = MoveSlot { id, pp: d.pp, maxpp: d.pp, disabled: false, hidden: false, used: false };
                }
                mon.n_moves = set.moves.len() as u8;
                b.sides[s].team[i] = mon;
            }
            b.sides[s].n = team.len() as u8;
            b.sides[s].pokemon_left = team.len() as u8;
        }

        // Team preview picks are queued as 'team' actions and sorted like any
        // others, so equal-speed Pokémon picked in the same slot draw from the RNG.
        let mut picks = [Battle::blank_action(ActKind::Team, 1); 2 * MAX_TEAM];
        let mut n = 0;
        for s in 0..2 {
            for i in 0..b.sides[s].n as usize {
                let r = MonRef { side: s as u8, idx: i as u8 };
                picks[n].priority = -(i as i32);
                picks[n].speed = b.action_speed(r);
                n += 1;
            }
        }
        b.speed_sort(&mut picks[..n], cmp_action, "team preview tie");

        b.queue.push(Battle::blank_action(ActKind::Start, 2));
        b.turn_loop();
        Ok(b)
    }

    /// Whether a move slot can be chosen (`Pokemon#getMoves`: PP left and not disabled).
    fn slot_usable(s: &MoveSlot) -> bool {
        s.pp > 0 && !s.disabled
    }

    fn usable_moves(&self, r: MonRef) -> bool {
        let m = self.mon(r);
        m.moves[..m.n_moves as usize].iter().any(Battle::slot_usable)
    }

    /// The target type Showdown's move request shows for a move
    /// (`Pokemon#getMoves`), which is what a choice of target is checked against.
    fn request_target(&self, r: MonRef, move_id: u16) -> Target {
        match move_id {
            // For anything but a Ghost, Curse is a move used on oneself.
            mv::CURSE if !self.has_type(r, Type::Ghost) => Target::User,
            // Heal Block keeps Pollen Puff off allies.
            mv::POLLENPUFF if self.vols(r).has(VolKind::Healblock) => Target::AdjacentFoe,
            _ => MOVES[move_id as usize].target,
        }
    }

    /// `Pokemon#isLastActive`: no living ally stands to its right.
    fn is_last_active(&self, r: MonRef) -> bool {
        let m = self.mon(r);
        m.is_active && (m.position as usize + 1..ACTIVE).all(|p| self.mon(self.active(r.side as usize, p)).fainted)
    }

    /// The one move choice of a Pokémon with no usable move, which Showdown
    /// turns into Struggle. Normally that is `move 1`. But Showdown does not
    /// tell a side's last active Pokémon which of its moves a foe's Imprison
    /// has sealed, so there it lists the real moves and wants the choice
    /// spelled like a use of the first one, target included.
    fn struggle_choice(&self, r: MonRef) -> Choice {
        let m = self.mon(r);
        let listed = self.is_last_active(r) && m.moves[..m.n_moves as usize].iter().any(|s| s.hidden && s.pp > 0);
        if listed {
            let t = self.request_target(r, m.moves[0].id);
            if t.is_chosen() {
                if let Some(loc) = [1i8, 2, -1, -2].into_iter().find(|&loc| self.valid_target_loc(loc, r, t)) {
                    return Choice::mv(0, loc);
                }
            }
        }
        Choice::mv(0, 0)
    }

    /// Every choice the given active slot may make at the current request.
    pub fn legal_choices(&self, side: usize, pos: usize) -> Vec<Choice> {
        let mut out = Vec::new();
        let r = self.active(side, pos);
        let m = self.mon(r);
        let s = &self.sides[side];
        let bench = |out: &mut Vec<Choice>| {
            for p in ACTIVE..s.n as usize {
                if !s.team[s.order[p] as usize].fainted {
                    out.push(Choice::Switch { to: p as u8 });
                }
            }
        };
        match self.request {
            Request::None => {}
            Request::Move => {
                if m.fainted {
                    out.push(Choice::Pass);
                    return out;
                }
                if m.locked_move != NO_MOVE {
                    // Locked into a move: Showdown lists that move alone, with no target to pick.
                    out.push(Choice::mv(0, 0));
                } else if !self.usable_moves(r) {
                    out.push(self.struggle_choice(r));
                } else {
                    // Any usable move can be combined with Mega Evolution.
                    let megas: &[bool] = if m.can_mega != NO_SPECIES { &[false, true] } else { &[false] };
                    for (i, slot) in m.moves[..m.n_moves as usize].iter().enumerate() {
                        if !Battle::slot_usable(slot) {
                            continue;
                        }
                        let t = self.request_target(r, slot.id);
                        for &mega in megas {
                            if t.is_chosen() {
                                for loc in [1i8, 2, -1, -2] {
                                    if self.valid_target_loc(loc, r, t) {
                                        out.push(Choice::Move { slot: i as u8, target: loc, mega });
                                    }
                                }
                            } else {
                                out.push(Choice::Move { slot: i as u8, target: 0, mega });
                            }
                        }
                    }
                }
                if m.trapped == Trapped::No {
                    bench(&mut out);
                }
            }
            Request::Switch => {
                if !m.switch_flag {
                    out.push(Choice::Pass);
                    return out;
                }
                if self.reviving(side, pos) {
                    // Revival Blessing: pick a fainted team member (one still in an active slot counts).
                    for p in 0..s.n as usize {
                        if s.team[s.order[p] as usize].fainted {
                            out.push(Choice::Switch { to: p as u8 });
                        }
                    }
                } else {
                    bench(&mut out);
                }
                if self.living_bench(side) < self.open_slots(side) {
                    // More empty slots than replacements: one of them stays empty.
                    out.push(Choice::Pass);
                }
            }
        }
        out
    }

    fn living_bench(&self, side: usize) -> usize {
        let s = &self.sides[side];
        (ACTIVE..s.n as usize).filter(|&p| !s.team[s.order[p] as usize].fainted).count()
    }

    /// Whether the Pokémon in this slot is being asked whom to revive.
    fn reviving(&self, side: usize, pos: usize) -> bool {
        self.sides[side].slot_conds[pos].has(SlotCond::Revivalblessing)
    }

    fn open_slots(&self, side: usize) -> usize {
        (0..ACTIVE).filter(|&p| self.mon(self.active(side, p)).switch_flag).count()
    }

    /// Whether one slot's choice is legal on its own (the allocation-free
    /// counterpart of `legal_choices`).
    fn slot_ok(&self, side: usize, pos: usize, c: Choice) -> bool {
        let r = self.active(side, pos);
        let m = self.mon(r);
        let s = &self.sides[side];
        let bench_ok =
            |to: u8| (ACTIVE..s.n as usize).contains(&(to as usize)) && !s.team[s.order[to as usize] as usize].fainted;
        match self.request {
            Request::None => false,
            Request::Move => match c {
                Choice::Pass => m.fainted,
                _ if m.fainted => false,
                Choice::Switch { to } => m.trapped == Trapped::No && bench_ok(to),
                Choice::Move { slot, target, mega } => {
                    if m.locked_move != NO_MOVE {
                        return c == Choice::mv(0, 0);
                    }
                    if !self.usable_moves(r) {
                        // Struggle; Showdown would ignore a Mega flag here, so it is not offered.
                        return c == self.struggle_choice(r);
                    }
                    if slot >= m.n_moves || !Battle::slot_usable(&m.moves[slot as usize]) {
                        return false;
                    }
                    if mega && m.can_mega == NO_SPECIES {
                        return false;
                    }
                    let t = self.request_target(r, m.moves[slot as usize].id);
                    if t.is_chosen() { target != 0 && self.valid_target_loc(target, r, t) } else { target == 0 }
                }
            },
            Request::Switch => match c {
                Choice::Pass => !m.switch_flag || self.living_bench(side) < self.open_slots(side),
                Choice::Switch { to } if self.reviving(side, pos) => {
                    m.switch_flag && to < s.n && s.team[s.order[to as usize] as usize].fainted
                }
                Choice::Switch { to } => m.switch_flag && bench_ok(to),
                Choice::Move { .. } => false,
            },
        }
    }

    /// The constraints that tie a side's two slots together: no two slots
    /// switching to the same Pokémon, only one Mega Evolution, and at a switch
    /// request every open slot filled while replacements last.
    #[inline(always)]
    fn pair_ok(&self, side: usize, c: &[Choice; ACTIVE]) -> bool {
        if let (Choice::Switch { to: a }, Choice::Switch { to: b }) = (c[0], c[1]) {
            if a == b {
                return false;
            }
        }
        if let (Choice::Move { mega: true, .. }, Choice::Move { mega: true, .. }) = (c[0], c[1]) {
            return false;
        }
        self.request != Request::Switch || self.switch_pair_ok(side, c)
    }

    /// The part of `pair_ok` that only a switch request needs.
    #[inline(never)]
    fn switch_pair_ok(&self, side: usize, c: &[Choice; ACTIVE]) -> bool {
        // Showdown counts down the switches and passes it still expects as it
        // reads the slots in order. A revival uses up a switch if one is left,
        // so with one replacement for a fainted Pokémon and a reviver to its
        // left, reviving and replacing cannot be chosen together (Showdown throws).
        let out = self.open_slots(side);
        let mut switches = out.min(self.living_bench(side));
        let mut passes = out - switches;
        for p in 0..ACTIVE {
            if !self.mon(self.active(side, p)).switch_flag {
                continue;
            }
            match c[p] {
                Choice::Pass => {
                    if passes == 0 {
                        return false;
                    }
                    passes -= 1;
                }
                Choice::Switch { .. } if self.reviving(side, p) => switches = switches.saturating_sub(1),
                Choice::Switch { .. } => {
                    if switches == 0 {
                        return false;
                    }
                    switches -= 1;
                }
                Choice::Move { .. } => return false,
            }
        }
        switches == 0
    }

    /// Whether a side's two slot choices are legal together.
    pub fn joint_ok(&self, side: usize, c: &[Choice; ACTIVE]) -> bool {
        (0..ACTIVE).all(|pos| self.slot_ok(side, pos, c[pos])) && self.pair_ok(side, c)
    }

    /// Every legal pair of slot choices for a side at the current request.
    pub fn joint_choices(&self, side: usize) -> Vec<[Choice; ACTIVE]> {
        let a = self.legal_choices(side, 0);
        let b = self.legal_choices(side, 1);
        let mut out = Vec::with_capacity(a.len() * b.len());
        for &x in &a {
            for &y in &b {
                let c = [x, y];
                if self.pair_ok(side, &c) {
                    out.push(c);
                }
            }
        }
        out
    }

    /// Submits both sides' choices and runs the battle to the next request (or the end).
    /// A side that is not being asked anything passes in both slots.
    pub fn choose(&mut self, choices: [[Choice; ACTIVE]; 2]) -> Result<(), Error> {
        if self.ended || self.request == Request::None {
            return Err(Error::BadChoice("the battle is not waiting for a choice".into()));
        }
        let request = self.request;
        for side in 0..2 {
            if !self.joint_ok(side, &choices[side]) {
                return Err(Error::BadChoice(format!("p{}: {:?} is not legal here", side + 1, choices[side])));
            }
        }

        // commitChoices
        self.update_speed();
        let old = self.queue;
        self.queue.clear();
        self.am_len = 0;
        for side in 0..2 {
            for pos in 0..ACTIVE {
                let r = self.active(side, pos);
                match choices[side][pos] {
                    Choice::Pass => {}
                    Choice::Move { slot, target, mega } => {
                        // A locked move goes where it was aimed. Struggle picks its
                        // own target, whatever the choice said.
                        let locked = self.mon(r).locked_move;
                        let (id, target) = if locked != NO_MOVE {
                            (locked, self.locked_move_loc(r, locked))
                        } else if self.usable_moves(r) {
                            (self.mon(r).moves[slot as usize].id, target)
                        } else {
                            (struggle_id(), 0)
                        };
                        // The move, and ahead of it whatever goes with it (a Mega Evolution, ...).
                        let (actions, n) = self.resolve_move_actions(r, id, target, mega && locked == NO_MOVE);
                        for a in &actions[..n] {
                            self.queue.push(*a);
                        }
                    }
                    Choice::Switch { to } if request == Request::Switch && self.reviving(side, pos) => {
                        // Not a switch at all: the chosen Pokémon is brought back.
                        let mut a = Battle::blank_action(ActKind::Revival, 6);
                        a.mon = Some(r);
                        a.switch_to = Some(self.active_at(side, to as usize));
                        self.mon_mut(r).switch_flag = false;
                        self.mon_mut(r).switch_move = NO_MOVE;
                        self.resolve_speed(&mut a);
                        self.queue.push(a);
                    }
                    Choice::Switch { to } => {
                        let incoming = self.active_at(side, to as usize);
                        let kind = if request == Request::Switch { ActKind::InstaSwitch } else { ActKind::Switch };
                        let a = self.resolve_switch(kind, r, incoming);
                        self.queue.push(a);
                    }
                }
            }
        }
        self.sort_queue();
        for a in old.as_slice() {
            self.queue.push(*a);
        }
        self.turn_loop();
        Ok(())
    }
}
