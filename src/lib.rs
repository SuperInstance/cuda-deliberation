//! # cuda-deliberation
//!
//! Consider/Resolve/Forfeit deliberation protocol.
//! Every proposal passes through agents who can accept, reject, or abstain.
//!
//! ```rust
//! use cuda_deliberation::{DeliberationEngine, Proposal, ProposalState};
//! use cuda_equipment::{Confidence, VesselId};
//!
//! let mut engine = DeliberationEngine::new(0.3); // forfeit gap threshold
//! let id = engine.propose("adopt_rust", VesselId(0), "Switch to Rust");
//! engine.consider(id, VesselId(1), Confidence::LIKELY);
//! engine.resolve(id, VesselId(1), true);
//! ```

pub use cuda_equipment::{Confidence, VesselId, FleetMessage, MessageType, Agent};

use std::collections::HashMap;

/// Lifecycle states of a proposal.
#[derive(Debug, Clone, PartialEq)]
pub enum ProposalState {
    Proposed,
    UnderConsideration,
    Accepted,
    Rejected,
    Forfeited,
    Expired,
}

/// A proposal being deliberated.
#[derive(Debug, Clone)]
pub struct Proposal {
    pub id: u64,
    pub title: String,
    pub description: String,
    pub proposer: VesselId,
    pub state: ProposalState,
    pub confidence: Confidence,
    pub votes_for: Vec<(VesselId, Confidence)>,
    pub votes_against: Vec<(VesselId, Confidence)>,
    pub abstentions: Vec<VesselId>,
    pub round: u32,
    pub created_at: u64,
    pub resolved_at: Option<u64>,
}

impl Proposal {
    pub fn new(id: u64, title: &str, description: &str, proposer: VesselId) -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        Self { id, title: title.to_string(), description: description.to_string(), proposer,
            state: ProposalState::Proposed, confidence: Confidence::HALF,
            votes_for: vec![], votes_against: vec![], abstentions: vec![],
            round: 0, created_at: SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64),
            resolved_at: None }
    }

    pub fn support_count(&self) -> usize { self.votes_for.len() }
    pub fn oppose_count(&self) -> usize { self.votes_against.len() }
    pub fn total_votes(&self) -> usize { self.votes_for.len() + self.votes_against.len() }

    pub fn consensus_ratio(&self) -> f64 {
        let total = self.total_votes();
        if total == 0 { return 0.5; }
        self.votes_for.len() as f64 / total as f64
    }

    pub fn has_voted(&self, vessel: VesselId) -> bool {
        self.votes_for.iter().any(|(v, _)| *v == vessel)
            || self.votes_against.iter().any(|(v, _)| *v == vessel)
            || self.abstentions.contains(&vessel)
    }
}

/// The deliberation engine — manages proposal lifecycle.
pub struct DeliberationEngine {
    proposals: HashMap<u64, Proposal>,
    next_id: u64,
    forfeit_gap: f64, // auto-forfeit when confidence gap exceeds this
    max_rounds: u32,
}

impl DeliberationEngine {
    pub fn new(forfeit_gap: f64) -> Self {
        Self { proposals: HashMap::new(), next_id: 1, forfeit_gap, max_rounds: 20 }
    }

    /// Create a new proposal.
    pub fn propose(&mut self, title: &str, proposer: VesselId, description: &str) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        let p = Proposal::new(id, title, description, proposer);
        self.proposals.insert(id, p);
        id
    }

    /// Agent considers a proposal (expresses interest).
    pub fn consider(&mut self, proposal_id: u64, vessel: VesselId, confidence: Confidence) -> Option<&ProposalState> {
        let p = self.proposals.get_mut(&proposal_id)?;
        if p.has_voted(vessel) { return Some(&p.state); }
        if p.state == ProposalState::Accepted || p.state == ProposalState::Rejected
            || p.state == ProposalState::Forfeited || p.state == ProposalState::Expired {
            return Some(&p.state);
        }
        p.state = ProposalState::UnderConsideration;
        p.round += 1;
        p.confidence = p.confidence.combine(confidence);
        Some(&p.state)
    }

    /// Agent resolves (votes on) a proposal.
    pub fn resolve(&mut self, proposal_id: u64, vessel: VesselId, accept: bool, confidence: Confidence) -> Option<ProposalAction> {
        let p = self.proposals.get_mut(&proposal_id)?;
        if p.has_voted(vessel) { return None; }
        if p.state == ProposalState::Accepted || p.state == ProposalState::Rejected
            || p.state == ProposalState::Forfeited {
            return None;
        }
        if accept {
            p.votes_for.push((vessel, confidence));
            p.confidence = p.confidence.combine(confidence);
        } else {
            p.votes_against.push((vessel, confidence));
            p.confidence = p.confidence.discount(confidence.value());
        }

        // Check auto-forfeit: if oppose confidence vastly exceeds support
        let support_conf: f64 = p.votes_for.iter().map(|(_, c)| c.value()).sum();
        let oppose_conf: f64 = p.votes_against.iter().map(|(_, c)| c.value()).sum();
        if oppose_conf > 0.0 && support_conf > 0.0 {
            let gap = oppose_conf / support_conf;
            if gap > 1.0 / self.forfeit_gap.max(0.01) {
                p.state = ProposalState::Forfeited;
                return Some(ProposalAction::Forfeited);
            }
        }

        // Check acceptance
        if p.consensus_ratio() >= 0.75 && p.votes_for.len() >= 2 {
            p.state = ProposalState::Accepted;
            return Some(ProposalAction::Accepted);
        }

        // Check rejection
        if p.consensus_ratio() <= 0.25 && p.votes_against.len() >= 2 {
            p.state = ProposalState::Rejected;
            return Some(ProposalAction::Rejected);
        }

        Some(ProposalAction::Continuing)
    }

    /// Agent forfeits (withdraws from consideration).
    pub fn forfeit(&mut self, proposal_id: u64, vessel: VesselId, reason: &str) -> Option<ProposalAction> {
        let p = self.proposals.get_mut(&proposal_id)?;
        p.abstentions.push(vessel);
        if p.confidence.value() < 0.2 {
            p.state = ProposalState::Forfeited;
            return Some(ProposalAction::Forfeited);
        }
        Some(ProposalAction::Continuing)
    }

    pub fn proposal(&self, id: u64) -> Option<&Proposal> { self.proposals.get(&id) }
    pub fn proposals(&self) -> Vec<&Proposal> { self.proposals.values().collect() }
    pub fn active_proposals(&self) -> Vec<&Proposal> {
        self.proposals.values()
            .filter(|p| matches!(p.state, ProposalState::Proposed | ProposalState::UnderConsideration))
            .collect()
    }

    /// Summary of all proposals.
    pub fn summary(&self) -> Vec<ProposalSummary> {
        self.proposals.values().map(|p| ProposalSummary {
            id: p.id, title: p.title.clone(), state: format!("{:?}", p.state),
            confidence: p.confidence.value(), support: p.votes_for.len(),
            oppose: p.votes_against.len(), round: p.round,
        }).collect()
    }
}

#[derive(Debug, Clone)]
pub enum ProposalAction {
    Accepted,
    Rejected,
    Forfeited,
    Continuing,
}

#[derive(Debug, Clone)]
pub struct ProposalSummary {
    pub id: u64,
    pub title: String,
    pub state: String,
    pub confidence: f64,
    pub support: usize,
    pub oppose: usize,
    pub round: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_propose() {
        let mut e = DeliberationEngine::new(0.3);
        let id = e.propose("test", VesselId(1), "A test proposal");
        let p = e.proposal(id).unwrap();
        assert_eq!(p.title, "test");
        assert!(matches!(p.state, ProposalState::Proposed));
    }

    #[test]
    fn test_consider_resolve() {
        let mut e = DeliberationEngine::new(0.3);
        let id = e.propose("adopt", VesselId(0), "Switch to Rust");
        e.consider(id, VesselId(1), Confidence::LIKELY);
        let action = e.resolve(id, VesselId(1), true, Confidence::SURE);
        assert!(matches!(action, Some(ProposalAction::Continuing)));
    }

    #[test]
    fn test_acceptance() {
        let mut e = DeliberationEngine::new(0.3);
        let id = e.propose("merge", VesselId(0), "Merge branch");
        e.resolve(id, VesselId(1), true, Confidence::SURE);
        e.resolve(id, VesselId(2), true, Confidence::SURE);
        let p = e.proposal(id).unwrap();
        assert_eq!(p.state, ProposalState::Accepted);
    }

    #[test]
    fn test_rejection() {
        let mut e = DeliberationEngine::new(0.3);
        let id = e.propose("bad", VesselId(0), "A bad idea");
        e.resolve(id, VesselId(1), false, Confidence::SURE);
        e.resolve(id, VesselId(2), false, Confidence::SURE);
        let p = e.proposal(id).unwrap();
        assert_eq!(p.state, ProposalState::Rejected);
    }

    #[test]
    fn test_forfeit() {
        let mut e = DeliberationEngine::new(0.3);
        let id = e.propose("drop", VesselId(0), "Drop it");
        e.forfeit(id, VesselId(1), "not interested");
        let action = e.resolve(id, VesselId(2), false, Confidence::SURE);
        assert!(matches!(action, Some(ProposalAction::Forfeited)));
    }

    #[test]
    fn test_no_double_vote() {
        let mut e = DeliberationEngine::new(0.3);
        let id = e.propose("vote", VesselId(0), "Test");
        e.resolve(id, VesselId(1), true, Confidence::SURE);
        let second = e.resolve(id, VesselId(1), false, Confidence::SURE);
        assert!(second.is_none());
    }

    #[test]
    fn test_summary() {
        let mut e = DeliberationEngine::new(0.3);
        e.propose("a", VesselId(0), "First");
        e.propose("b", VesselId(0), "Second");
        let s = e.summary();
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn test_active_proposals() {
        let mut e = DeliberationEngine::new(0.3);
        e.propose("active", VesselId(0), "Active proposal");
        let id2 = e.propose("reject", VesselId(0), "Will reject");
        e.resolve(id2, VesselId(1), false, Confidence::SURE);
        e.resolve(id2, VesselId(2), false, Confidence::SURE);
        assert_eq!(e.active_proposals().len(), 1);
    }
}
