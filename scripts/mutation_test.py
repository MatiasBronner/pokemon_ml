#!/usr/bin/env python3
"""Mutation test: does the comparison with Showdown have teeth?

Injects one small bug at a time into a copy of the engine (a wrong multiplier,
an inverted condition, a dropped line), rebuilds, replays recorded Showdown
battles and reports whether the replay noticed. A bug that goes unnoticed is
either an equivalent change or a gap in what the recorded battles exercise.

    scripts/mutation_test.py CASES.jsonl [MORE.jsonl ...] [--only TEXT]

Record a few thousand battles first (oracle/gen_cases.js); small corpora miss
the rarer effects. Each mutation is a (name, file, old text, new text) entry;
when the source changes under an entry it is reported as BAD PATTERN and
should be updated rather than deleted.
"""
import os
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(REPO, 'target', 'mutation')
A, I, B, M, C, E = 'src/abilities.rs', 'src/items.rs', 'src/battle.rs', 'src/moves.rs', 'src/conditions.rs', 'src/events.rs'
MUT = [
 # --- abilities
 ('hugepower x2 -> x1.5', A, "(ab::HUGEPOWER | ab::PUREPOWER, Ev::ModifyAtk, Pre::On) => self.chain_modify(2, 1),", "(ab::HUGEPOWER | ab::PUREPOWER, Ev::ModifyAtk, Pre::On) => self.chain_modify(3, 2),"),
 ('ironfist 4915 -> 5325', A, "ab::IRONFIST => (F_PUNCH, 4915),", "ab::IRONFIST => (F_PUNCH, 5325),"),
 ('megalauncher 1.5 -> 1.3', A, "ab::MEGALAUNCHER => (F_PULSE, 6144),", "ab::MEGALAUNCHER => (F_PULSE, 5325),"),
 ('strongjaw flag', A, "ab::STRONGJAW => (F_BITE, 6144),", "ab::STRONGJAW => (F_PUNCH, 6144),"),
 ('toughclaws 5325 -> 4915', A, "ab::TOUGHCLAWS => (F_CONTACT, 5325),", "ab::TOUGHCLAWS => (F_CONTACT, 4915),"),
 ('sharpness flag', A, "ab::SHARPNESS => (F_SLICING, 6144),", "ab::SHARPNESS => (F_BITE, 6144),"),
 ('technician threshold 60 -> 70', A, "<= 60 {", "<= 70 {"),
 ('blaze type', A, "ab::BLAZE => Type::Fire,", "ab::BLAZE => Type::Water,"),
 ('pinch threshold 1/3 -> 1/2', A, "m.hp as u32 * 3 <= m.max_hp() as u32", "m.hp as u32 * 2 <= m.max_hp() as u32"),
 ('adaptability 2 -> 1.75', A, "{ 9216 } else { 8192 }", "{ 9216 } else { 7168 }"),
 # ("analytic counts itself" is an equivalent change: the user's own action has already left the queue.)
 ('analytic 1.3 -> 1.2', A, "if boosted {\n                    return self.chain_modify(5325, 4096);", "if boosted {\n                    return self.chain_modify(4915, 4096);"),
 ('analytic never boosted', A, "let boosted = !actives[..n]", "let boosted = actives[..n]"),
 ('merciless ratio', A, "return Res::Num(5);", "return Res::Num(2);"),
 ('superluck +1 -> +2', A, "(ab::SUPERLUCK, Ev::ModifyCritRatio, Pre::On) => Res::Num(relay.num() + 1),", "(ab::SUPERLUCK, Ev::ModifyCritRatio, Pre::On) => Res::Num(relay.num() + 2),"),
 ('stall sign', A, "(ab::STALL, Ev::FractionalPriority, Pre::On) => Res::Num(-1),", "(ab::STALL, Ev::FractionalPriority, Pre::On) => Res::Num(1),"),
 ('quickfeet 1.5 -> 2', A, "(ab::QUICKFEET, Ev::ModifySpe, Pre::On) => {\n                if self.mon(holder).status != Status::None {\n                    return self.chain_modify(6144, 4096);", "(ab::QUICKFEET, Ev::ModifySpe, Pre::On) => {\n                if self.mon(holder).status != Status::None {\n                    return self.chain_modify(8192, 4096);"),
 ('stakeout inverted', A, "self.mon(d).active_turns == 0", "self.mon(d).active_turns > 0"),
 ('unaware keeps evasion', A, "                    self.event.boosts[EVA] = 0;\n", ""),
 ('unaware keeps attack', A, "                    self.event.boosts[ATK] = 0;\n                    self.event.boosts[DEF] = 0;\n                    self.event.boosts[SPA] = 0;", "                    self.event.boosts[DEF] = 0;\n                    self.event.boosts[SPA] = 0;"),
 ('hustle atk', A, "modify(relay.num() as u32, 6144)", "modify(relay.num() as u32, 5325)"),
 ('hustle accuracy', A, "return self.chain_modify(3277, 4096);", "return self.chain_modify(3686, 4096);"),
 ('keeneye ignores nothing', A, "self.am[m as usize].ignore_evasion = true;", "self.am[m as usize].ignore_evasion = false;"),
 ('contrary off', A, "*b = -*b;", "*b = *b;"),
 ('magicguard only recoil', A, "(ab::MAGICGUARD, Ev::Damage, Pre::On) => {\n                if !e.effect.is_move() {", "(ab::MAGICGUARD, Ev::Damage, Pre::On) => {\n                if e.effect == Eff::Recoil {"),
 ('battlearmor off', A, "(ab::BATTLEARMOR | ab::SHELLARMOR, Ev::CriticalHit, Pre::On) => FALSE,", "(ab::BATTLEARMOR | ab::SHELLARMOR, Ev::CriticalHit, Pre::On) => Res::Undef,"),
 ('-ate boost 1.2 -> 1.3', A, "if self.am[m as usize].type_changer_boosted == Eff::Ability(ability) {\n                        return self.chain_modify(4915, 4096);", "if self.am[m as usize].type_changer_boosted == Eff::Ability(ability) {\n                        return self.chain_modify(5325, 4096);"),
 ('pixilate type', A, "ab::PIXILATE => Type::Fairy,", "ab::PIXILATE => Type::Ice,"),
 ('fluffy contact /2 -> /4', A, "                    num /= 2;", "                    num /= 4;"),
 ('heatproof burn /2 -> /3', A, "return Res::Num(relay.num() / 2);", "return Res::Num(relay.num() / 3);"),
 ('stickyhold off', A, "if e.source.is_some() && e.source != Some(holder) {\n                    return FALSE;", "if e.source.is_some() && e.source != Some(holder) {\n                    return Res::Undef;"),
 ('minus/plus counts itself', A, "if a != holder && (self.has_ability(a, ab::MINUS)", "if (self.has_ability(a, ab::MINUS)"),
 ('firemane type', A, "(ab::FIREMANE, Ev::ModifyAtk | Ev::ModifySpA, Pre::On) => {\n                if mtype == Some(Type::Fire) {", "(ab::FIREMANE, Ev::ModifyAtk | Ev::ModifySpA, Pre::On) => {\n                if mtype == Some(Type::Water) {"),
 ('intimidate -1 -> -2', A, "self.boost1(ATK, -1, Some(f), Some(holder), Eff::None);", "self.boost1(ATK, -2, Some(f), Some(holder), Eff::None);"),
 ('sturdy leaves 2 hp', A, "return Res::Num(m.hp as i32 - 1);", "return Res::Num(m.hp as i32 - 2);"),
 ('regenerator 1/3 -> 1/4', A, "let amount = self.mon(holder).max_hp() as i32 / 3;", "let amount = self.mon(holder).max_hp() as i32 / 4;"),
 ('moody +2 -> +1', A, "[(plus, 2), (minus, -1)]", "[(plus, 1), (minus, -1)]"),
 ('synchronize passes freeze', A, "matches!(e.status, Status::Slp | Status::Frz)", "matches!(e.status, Status::Slp)"),
 ('wandering spirit contact inverted', A, "(ab::WANDERINGSPIRIT, Ev::DamagingHit, Pre::On) => {\n                if let (Some(m), Some(target), Some(source)) = (mi, e.target, e.source) {\n                    if self.makes_contact(m) {", "(ab::WANDERINGSPIRIT, Ev::DamagingHit, Pre::On) => {\n                if let (Some(m), Some(target), Some(source)) = (mi, e.target, e.source) {\n                    if !self.makes_contact(m) {"),
 ('shield dust off', A, "self.event.secs &= !(1 << k);", "self.event.secs &= 0xff;"),
 ('skill link min hits', A, "am.multihit = (am.multihit.1, am.multihit.1);", "am.multihit = (am.multihit.0, am.multihit.0);"),
 ('serene grace off', A, "s.chance *= 2;", "s.chance *= 1;"),
 ('sheer force no boost', A, "am.has_sheer_force = true;", "am.has_sheer_force = false;"),
 ('sniper 1.5 -> 2', A, "if self.am[m as usize].hit_data[self.slot_index(target)].crit {\n                        return self.chain_modify(6144, 4096);", "if self.am[m as usize].hit_data[self.slot_index(target)].crit {\n                        return self.chain_modify(8192, 4096);"),
 ('multiscale 0.5 -> 0.75', A, "if m.hp >= m.max_hp() {\n                    return self.chain_modify(2048, 4096);", "if m.hp >= m.max_hp() {\n                    return self.chain_modify(3072, 4096);"),
 ('friend guard 0.75 -> 0.5', A, "if target != holder && self.is_ally(target, holder) {\n                        return self.chain_modify(3072, 4096);", "if target != holder && self.is_ally(target, holder) {\n                        return self.chain_modify(2048, 4096);"),
 ('thick fat only ice', A, "matches!(mtype, Some(Type::Ice | Type::Fire))", "matches!(mtype, Some(Type::Ice))"),
 ('rivalry swapped', A, "return if a == d {", "return if a != d {"),
 ('supreme overlord table', A, "[4096, 4506, 4915, 5325, 5734, 6144]", "[4096, 4915, 4915, 5325, 5734, 6144]"),
 ('shadow tag off', A, "self.try_trap(pokemon, true);", ""),
 ('pressure 1 -> 2', A, "                Res::Num(1)\n            }\n\n            // ---- Punk Rock", "                Res::Num(2)\n            }\n\n            // ---- Punk Rock"),
 ('opportunist copies nothing', A, "m.ability_boosts[k] += e.boosts[k];", "m.ability_boosts[k] += 0;"),
 ('trace picks first', A, "let target = cands[self.rand(k as u32, \"trace\") as usize];", "self.rand(k as u32, \"trace\");\n                let target = cands[0];"),
 ('filter 0.75 -> 0.5', A, "(ab::FILTER | ab::SOLIDROCK, Ev::ModifyDamage, Pre::Source) => {\n                if let (Some(m), Some(target)) = (mi, e.source) {\n                    if self.am[m as usize].hit_data[self.slot_index(target)].type_mod > 0 {\n                        return self.chain_modify(3072, 4096);", "(ab::FILTER | ab::SOLIDROCK, Ev::ModifyDamage, Pre::Source) => {\n                if let (Some(m), Some(target)) = (mi, e.source) {\n                    if self.am[m as usize].hit_data[self.slot_index(target)].type_mod > 0 {\n                        return self.chain_modify(2048, 4096);"),
 ('tangled feet 0.5 -> 0.75', A, "self.mon(holder).volatiles.has(VolKind::Confusion) {\n                    return self.chain_modify(2048, 4096);", "self.mon(holder).volatiles.has(VolKind::Confusion) {\n                    return self.chain_modify(3072, 4096);"),
 ('aura guard 0.5 -> 0.75', A, "(ab::AURAGUARD, Ev::ModifyDamage, Pre::Source) => {\n                if mflags & F_CONTACT != 0 {\n                    return self.chain_modify(2048, 4096);", "(ab::AURAGUARD, Ev::ModifyDamage, Pre::Source) => {\n                if mflags & F_CONTACT != 0 {\n                    return self.chain_modify(3072, 4096);"),
 ('fairy aura 5448 -> 5325', A, "self.chain_modify(5448, 4096)", "self.chain_modify(5325, 4096)"),
 ('long reach keeps contact', A, "self.am[m as usize].flags &= !F_CONTACT;", "self.am[m as usize].flags &= !F_SOUND;"),
 ('no guard accuracy', A, "                    return TRUE;\n                }\n                relay\n", "                    return relay;\n                }\n                relay\n"),
 ('galewings any hp', A, "if mtype == Some(Type::Flying) && m.hp == m.max_hp() {", "if mtype == Some(Type::Flying) {"),
 ('prankster +1 -> +2', A, "self.am[m as usize].prankster_boosted = true;\n                    return Res::Num(relay.num() + 1);", "self.am[m as usize].prankster_boosted = true;\n                    return Res::Num(relay.num() + 2);"),
 ('quick draw odds', A, "self.chance(3, 10, \"quick draw\")", "self.chance(4, 10, \"quick draw\")"),
 ('steely spirit 1.5 -> 1.3', A, "(ab::STEELYSPIRIT, Ev::BasePower, Pre::Ally) => {\n                if mtype == Some(Type::Steel) {\n                    return self.chain_modify(6144, 4096);", "(ab::STEELYSPIRIT, Ev::BasePower, Pre::Ally) => {\n                if mtype == Some(Type::Steel) {\n                    return self.chain_modify(5325, 4096);"),
 ('reckless 1.2 -> 1.3', A, "self.am[m as usize].d().recoil.0 > 0) {\n                    return self.chain_modify(4915, 4096);", "self.am[m as usize].d().recoil.0 > 0) {\n                    return self.chain_modify(5325, 4096);"),
 ('scrappy off', A, "self.am[m as usize].ignore_immunity = IgnoreImm::NormalFighting;", "self.am[m as usize].ignore_immunity = IgnoreImm::No;"),
 ('liquid voice type', A, "self.am[m as usize].typ = Type::Water;", "self.am[m as usize].typ = Type::Ice;"),
 ('marvel scale 1.5 -> 2', A, "(ab::MARVELSCALE, Ev::ModifyDef, Pre::On) => {\n                if self.mon(holder).status != Status::None {\n                    return self.chain_modify(6144, 4096);", "(ab::MARVELSCALE, Ev::ModifyDef, Pre::On) => {\n                if self.mon(holder).status != Status::None {\n                    return self.chain_modify(8192, 4096);"),
 ('fur coat 2 -> 1.5', A, "(ab::FURCOAT, Ev::ModifyDef, Pre::On) => self.chain_modify(2, 1),", "(ab::FURCOAT, Ev::ModifyDef, Pre::On) => self.chain_modify(3, 2),"),
 ('guts 1.5 -> 2', A, "(ab::GUTS, Ev::ModifyAtk, Pre::On) => {\n                if self.mon(holder).status != Status::None {\n                    return self.chain_modify(6144, 4096);", "(ab::GUTS, Ev::ModifyAtk, Pre::On) => {\n                if self.mon(holder).status != Status::None {\n                    return self.chain_modify(8192, 4096);"),
 ('compound eyes 1.3 -> 1.2', A, "(ab::COMPOUNDEYES, Ev::ModifyAccuracy, Pre::Source) => {\n                if matches!(relay, Res::Num(_)) {\n                    return self.chain_modify(5325, 4096);", "(ab::COMPOUNDEYES, Ev::ModifyAccuracy, Pre::Source) => {\n                if matches!(relay, Res::Num(_)) {\n                    return self.chain_modify(4915, 4096);"),
 ('queenly majesty blocks nothing', A, "if self.is_ally(aimed_at, holder) && self.am[m as usize].priority > 0 {\n                    return FALSE;", "if self.is_ally(aimed_at, holder) && self.am[m as usize].priority > 0 {\n                    return Res::Undef;"),
 ('stench/kings rock 10% -> 20%', I, "                chance: 10,", "                chance: 20,"),
 # --- abilities checked directly in the simulator core
 ('prankster hits Dark types', M, "&& !self.type_allows(t, 7)", "&& false"),
 ('parental bond second hit 0.25 -> 0.5', M, "dmg = modify(dmg, 1024);\n        }\n        // WeatherModifyDamage", "dmg = modify(dmg, 2048);\n        }\n        // WeatherModifyDamage"),
 ('guts still halved by burn', M, "            && !self.has_ability(user, ab::GUTS)\n", ""),
 ('klutz off', B, "ITEMS[m.item as usize].flags & IF_IGNORE_KLUTZ == 0 && self.has_ability(r, ab::KLUTZ)", "false"),
 ('corrosion off', B, "if !corrosive && !self.run_status_immunity(r, Imm::Status(status)) {", "if !self.run_status_immunity(r, Imm::Status(status)) {"),
 ('levitate off', B, "(self.has_ability(r, ab::LEVITATE) || self.has_ability(r, ab::EELEVATE))", "self.has_ability(r, ab::EELEVATE)"),
 ('stalwart off', B, "if self.has_ability(user, ab::STALWART) || self.has_ability(user, ab::PROPELLERTAIL) {", "if false {"),
 ('early bird off', C, "                if early {", "                if early && false {"),
 ('quick feet still halved by paralysis', C, "if !self.has_ability(holder, ab::QUICKFEET) {", "if true {"),
 ('mold breaker off', E, "if ABILITIES[a as usize].flags & AF_BREAKABLE != 0 && self.suppressing_ability(Some(h.holder)) {", "if false {"),
 ('sheer force keeps life orb recoil', M, "        if !self.suppressing_secondaries() {", "        if true {"),
 ('flash fire boost 1.5 -> 2', C, "if fire && self.has_ability(holder, ab::FLASHFIRE) {\n                    return self.chain_modify(6144, 4096);", "if fire && self.has_ability(holder, ab::FLASHFIRE) {\n                    return self.chain_modify(8192, 4096);"),
 ('unburden x2 -> x1.5', C, "!self.ignoring_ability(holder) {\n                    return self.chain_modify(2, 1);", "!self.ignoring_ability(holder) {\n                    return self.chain_modify(3, 2);"),
 # --- items
 ('charcoal type', I, "it::CHARCOAL => Type::Fire,", "it::CHARCOAL => Type::Water,"),
 ('type booster 4915 -> 5324', I, "if self.event_move().is_some_and(|mi| self.am[mi as usize].typ == t) {\n                return self.chain_modify(4915, 4096);", "if self.event_move().is_some_and(|mi| self.am[mi as usize].typ == t) {\n                return self.chain_modify(5324, 4096);"),
 ('resist berry 0.5 -> 0.75', I, "if self.eat_item(target, false, None, Eff::None) {\n                            return self.chain_modify(2048, 4096);", "if self.eat_item(target, false, None, Eff::None) {\n                            return self.chain_modify(3072, 4096);"),
 ('chilan needs super effective', I, "(item == it::CHILANBERRY || am.hit_data[slot].type_mod > 0)", "(am.hit_data[slot].type_mod > 0)"),
 ('expert belt 4915 -> 5324', I, "if self.am[mi as usize].hit_data[slot].type_mod > 0 {\n                        return self.chain_modify(4915, 4096);", "if self.am[mi as usize].hit_data[slot].type_mod > 0 {\n                        return self.chain_modify(5324, 4096);"),
 ('muscle band 4505 -> 4915', I, "Category::Physical) {\n                    return self.chain_modify(4505, 4096);", "Category::Physical) {\n                    return self.chain_modify(4915, 4096);"),
 ('wise glasses 4505 -> 4915', I, "Category::Special) {\n                    return self.chain_modify(4505, 4096);", "Category::Special) {\n                    return self.chain_modify(4915, 4096);"),
 ('scope lens +1 -> +2', I, "(it::SCOPELENS, Ev::ModifyCritRatio, Pre::On) => Res::Num(relay.num() + 1),", "(it::SCOPELENS, Ev::ModifyCritRatio, Pre::On) => Res::Num(relay.num() + 2),"),
 ('wide lens 4505 -> 4915', I, "(it::WIDELENS, Ev::ModifyAccuracy, Pre::Source) => {\n                if matches!(relay, Res::Num(_)) {\n                    return self.chain_modify(4505, 4096);", "(it::WIDELENS, Ev::ModifyAccuracy, Pre::Source) => {\n                if matches!(relay, Res::Num(_)) {\n                    return self.chain_modify(4915, 4096);"),
 ('zoom lens inverted', I, "e.target.is_some_and(|t| !self.will_move(t))", "e.target.is_some_and(|t| self.will_move(t))"),
 ('bright powder 3686 -> 3277', I, "return self.chain_modify(3686, 4096);", "return self.chain_modify(3277, 4096);"),
 ('choice scarf 1.5 -> 1.3', I, "(it::CHOICESCARF, Ev::ModifySpe, Pre::On) => self.chain_modify(6144, 4096),", "(it::CHOICESCARF, Ev::ModifySpe, Pre::On) => self.chain_modify(5325, 4096),"),
 ('iron ball speed 0.5 -> 0.75', I, "(it::IRONBALL, Ev::ModifySpe, Pre::On) => self.chain_modify(2048, 4096),", "(it::IRONBALL, Ev::ModifySpe, Pre::On) => self.chain_modify(3072, 4096),"),
 ('iron ball does not ground', B, "if item == it::IRONBALL {", "if false {"),
 ('air balloon does not lift', B, "item != it::AIRBALLOON", "true"),
 ('leek +2 -> +1', I, "return Res::Num(relay.num() + 2);", "return Res::Num(relay.num() + 1);"),
 ('light ball species', I, 'if base == "pikachu" {', 'if base == "raichu" {'),
 ('life orb 5324 -> 5325', I, "(it::LIFEORB, Ev::ModifyDamage, Pre::On) => self.chain_modify(5324, 4096),", "(it::LIFEORB, Ev::ModifyDamage, Pre::On) => self.chain_modify(5325, 4096),"),
 ('life orb recoil 1/10 -> 1/8', I, "let d = div1(self.mon(source).max_hp() as u32, 10);", "let d = div1(self.mon(source).max_hp() as u32, 8);"),
 ('metronome first step', C, "[4096, 4915, 5734, 6553, 7372, 8192]", "[4096, 5000, 5734, 6553, 7372, 8192]"),
 ('big root 1.3 -> 1.5', I, "if e.effect == Eff::Drain {\n                    return self.chain_modify(5324, 4096);", "if e.effect == Eff::Drain {\n                    return self.chain_modify(6144, 4096);"),
 ('quick claw odds', I, 'self.chance(1, 5, "quick claw")', 'self.chance(1, 4, "quick claw")'),
 ('focus band odds', I, 'self.chance(1, 10, "focus band")', 'self.chance(1, 9, "focus band")'),
 ('focus sash leaves 2 hp', I, "if self.use_item(target, None, Eff::None) {\n                        return Res::Num(self.mon(target).hp as i32 - 1);", "if self.use_item(target, None, Eff::None) {\n                        return Res::Num(self.mon(target).hp as i32 - 2);"),
 ('shed shell traps', I, "(it::SHEDSHELL, Ev::TrapPokemon, Pre::On) => {\n                self.mon_mut(holder).trapped = Trapped::No;", "(it::SHEDSHELL, Ev::TrapPokemon, Pre::On) => {\n                self.mon_mut(holder).trapped = Trapped::Yes;"),
 ('leppa 10 -> 5', I, "{ 20 } else { 10 }", "{ 20 } else { 5 }"),
 ('shell bell 1/8 -> 1/4', I, "div1(total as u32, 8)", "div1(total as u32, 4)"),
 ('rocky helmet 1/6 -> 1/8', I, "div1(self.mon(source).max_hp() as u32, 6)", "div1(self.mon(source).max_hp() as u32, 8)"),
 ('leftovers 1/16 -> 1/8', I, "(it::LEFTOVERS, Ev::Residual, Pre::On) => {\n                let amount = div1(self.mon(holder).max_hp() as u32, 16);", "(it::LEFTOVERS, Ev::Residual, Pre::On) => {\n                let amount = div1(self.mon(holder).max_hp() as u32, 8);"),
 ('gem 1.3 -> 1.5', C, "(VolKind::Gem, Ev::BasePower) => self.chain_modify(5325, 4096),", "(VolKind::Gem, Ev::BasePower) => self.chain_modify(6144, 4096),"),
 ('white herb sets +1', I, "                    if *b < 0 {\n                        *b = 0;", "                    if *b < 0 {\n                        *b = 1;"),
 ('persim does nothing', I, "(it::PERSIMBERRY, Ev::Eat, Pre::On) => {\n                self.remove_volatile(holder, VolKind::Confusion);", "(it::PERSIMBERRY, Ev::Eat, Pre::On) => {"),
 ('sitrus 1/4 -> 1/3', I, "let amount = if item == it::ORANBERRY { 10 } else { div1(self.mon(holder).max_hp() as u32, 4) };\n                self.heal(", "let amount = if item == it::ORANBERRY { 10 } else { div1(self.mon(holder).max_hp() as u32, 3) };\n                self.heal("),
 ('choice lock off', C, "                    if m.moves[k].id + 1 != locked {\n                        m.moves[k].disabled = true;", "                    if m.moves[k].id + 1 != locked {\n                        m.moves[k].disabled = false;"),
]

def sh(cmd, **kw):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)

def main():
    args = sys.argv[1:]
    only = None
    if '--only' in args:
        i = args.index('--only'); only = args[i + 1]; del args[i:i + 2]
    corpora = [os.path.abspath(c) for c in args]
    if not corpora:
        sys.exit(__doc__)
    os.makedirs(WORK, exist_ok=True)
    for d in ('src', 'tests'):
        shutil.rmtree(os.path.join(WORK, d), ignore_errors=True)
        shutil.copytree(os.path.join(REPO, d), os.path.join(WORK, d), copy_function=shutil.copy)
    for f in ('Cargo.toml', 'Cargo.lock', 'rustfmt.toml'):
        shutil.copy(os.path.join(REPO, f), os.path.join(WORK, f))
    shutil.rmtree(os.path.join(WORK, 'examples'), ignore_errors=True)
    shutil.copytree(os.path.join(REPO, 'examples'), os.path.join(WORK, 'examples'), copy_function=shutil.copy)
    results = []
    # baseline must pass
    r = sh('cargo build --release --bin difftest 2>&1 | tail -1', cwd=WORK)
    for c in corpora:
        r = sh(f'./target/release/difftest {c} --quiet', cwd=WORK)
        if r.returncode != 0:
            print('BASELINE FAILS on', c, r.stdout[-300:]); return
    print('baseline passes on', len(corpora), 'corpora', flush=True)
    for name, f, old, new in MUT:
        if only and only not in name:
            continue
        path = os.path.join(WORK, f)
        src = open(path).read()
        n = src.count(old)
        if n != 1:
            print(f'BAD PATTERN ({n} matches): {name}', flush=True); results.append((name, 'bad')); continue
        open(path, 'w').write(src.replace(old, new))
        b = sh('cargo build --release --bin difftest 2>&1 | grep -E "^error" -A6', cwd=WORK)
        if b.stdout.strip():
            print(f'DOES NOT COMPILE: {name}\n{b.stdout[:400]}', flush=True); results.append((name, 'nocompile'))
        else:
            caught = None
            for c in corpora:
                r = sh(f'./target/release/difftest {c} --quiet', cwd=WORK)
                if r.returncode != 0:
                    line = [l for l in r.stdout.splitlines() if 'diverged' in l or 'panicked' in l]
                    caught = (os.path.basename(c), line[-1] if line else 'crash')
                    break
            if caught:
                print(f'caught   {name:<44} {caught[0]}: {caught[1][:90]}', flush=True); results.append((name, 'caught'))
            else:
                print(f'MISSED   {name}', flush=True); results.append((name, 'missed'))
        open(path, 'w').write(src)
    tally = {}
    for _, r in results: tally[r] = tally.get(r, 0) + 1
    print('summary:', tally)
    print('missed:', [n for n, r in results if r == 'missed'])

main()
