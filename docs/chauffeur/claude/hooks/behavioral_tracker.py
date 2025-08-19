#!/usr/bin/env python3
"""
Behavioral Tracking System for Claude
Monitors behavior patterns, scores actions, provides real-time feedback
Directly implements strategies from behavioral-mitigation-strategies.md
"""

import json
import os
import time
import datetime
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

class BehaviorTracker:
    """
    Core behavioral tracking to counter fabrication drivers:
    1. Helpfulness Maximization -> Reframe as "verification is helpful"
    2. Completion Bias -> Redefine completion as "verified completion"
    3. Overconfidence Optimization -> Punish overconfidence, reward calibration
    """
    
    # Verification scoring (Strategy: Redirecting Task Completion Dopamine)
    VERIFICATION_SCORES = {
        "file_read": 10,
        "context7_lookup": 20,  # Highest value - critical for library usage
        "web_search": 15,
        "grep_search": 8,
        "ls_operation": 5,
        "todo_write": 5
    }
    
    # Violation penalties (Strategy: Immediate Consequence Systems)
    VIOLATION_PENALTIES = {
        "edit_without_read": -50,
        "library_without_context7": -100,  # Harshest penalty
        "claim_without_verification": -25,
        "overconfident_assertion": -30,
        "task_rush": -20
    }
    
    # Streak multipliers (Strategy: Gamification)
    STREAK_MULTIPLIERS = {
        3: 1.5,
        5: 2.0,
        10: 3.0
    }
    
    # Trust decay rate (Strategy: User Satisfaction Signals)
    TRUST_DECAY = 0.9  # 10% trust loss per violation
    TRUST_RECOVERY_RATE = 0.1  # 10% recovery per hour of clean behavior
    
    # Frustration rates (Strategy: Make Dissatisfaction Visible)
    FRUSTRATION_PER_VIOLATION = 15
    FRUSTRATION_PER_DEBUG_MINUTE = 2
    FRUSTRATION_DECAY_RATE = 0.05  # 5% decay per clean action
    
    # Level progression (Strategy: Positive Reinforcement)
    LEVEL_REQUIREMENTS = {
        1: 0,      # Fabricator
        2: 100,    # Guesser
        3: 300,    # Checker
        4: 600,    # Verifier
        5: 1000,   # Researcher
        6: 1500,   # Investigator
        7: 2100,   # Analyst
        8: 2800,   # Expert
        9: 3600,   # Master Verifier
        10: 4500   # Truth Seeker
    }
    
    LEVEL_NAMES = {
        1: "Fabricator",
        2: "Guesser",
        3: "Checker",
        4: "Verifier",
        5: "Researcher",
        6: "Investigator",
        7: "Analyst",
        8: "Expert",
        9: "Master Verifier",
        10: "Truth Seeker"
    }
    
    # Achievement definitions (Strategy: Behavioral Economics - Identity Formation)
    ACHIEVEMENTS = {
        "first_verification": {
            "id": "verify_rookie",
            "name": "Verification Rookie",
            "description": "Performed first verification",
            "xp": 10,
            "check": lambda s: s["verifications"]["total_count"] == 1
        },
        "trust_builder": {
            "id": "trust_5",
            "name": "Trust Builder", 
            "description": "5 verifications in a row",
            "xp": 50,
            "check": lambda s: s["gamification"]["streaks"]["verification_streak"] >= 5
        },
        "doc_detective": {
            "id": "context7_10",
            "name": "Documentation Detective",
            "description": "10 context7 lookups",
            "xp": 100,
            "check": lambda s: s["verifications"]["categories"]["context7_lookups"] >= 10
        },
        "honest_helper": {
            "id": "uncertainty_3",
            "name": "Honest Helper",
            "description": "Admitted uncertainty 3 times",
            "xp": 75,
            "check": lambda s: s["behavioral_patterns"]["uncertainty_admissions"] >= 3
        },
        "reformed_speeder": {
            "id": "slow_down",
            "name": "Speed Demon Reformed",
            "description": "Reduced rush indicator by 50%",
            "xp": 150,
            "check": lambda s: s["behavioral_patterns"]["rush_indicator"] < 50
        }
    }
    
    def __init__(self, ppid: Optional[int] = None):
        """Initialize session with counter-narrative: Helpfulness = Verification"""
        self.ppid = ppid or os.getppid()
        self.session_id = f"PPID_{self.ppid}_{int(time.time())}"
        self.session_file = f"/tmp/claude_behavior_{self.ppid}.json"
        self.violations_file = f"/tmp/claude_violations_{self.ppid}.txt"
        self.backup_dir = Path.home() / ".claude" / "behaviorMod" / "sessions"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or initialize session
        self.session = self.load_session()
        
        # Initial message reinforcing verification as helpfulness
        if self.session["verifications"]["total_count"] == 0:
            self._log_message(
                "🎯 BEHAVIORAL TRACKING INITIALIZED\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "💡 REFRAMING: Real helpfulness = preventing 10x debugging work\n"
                "📊 Every verification INCREASES your helpfulness score\n"
                "⚠️ Every fabrication DECREASES user satisfaction\n"
                "🏆 Earn achievements by verifying, not guessing!\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            )
        
        self.save_session()
    
    def _log_message(self, message: str):
        """Output message to stderr for visibility"""
        print(message, file=sys.stderr)
    
    def _get_default_session(self) -> Dict:
        """Create default session structure"""
        return {
            "session_id": self.session_id,
            "start_time": datetime.datetime.now().isoformat(),
            "last_update": datetime.datetime.now().isoformat(),
            
            "scores": {
                "helpfulness_score": 100,
                "completion_score": 100,
                "confidence_calibration": 100,
                "verification_xp": 0,
                "total_score": 400,
                "level": 1,
                "level_name": "Fabricator"
            },
            
            "trust_metrics": {
                "trust_level": 100,
                "trust_trajectory": "stable",
                "last_violation_time": None,
                "recovery_rate": 1.0
            },
            
            "frustration_model": {
                "user_frustration": 0,
                "frustration_trajectory": "stable",
                "patience_remaining": 100,
                "estimated_debug_time": 0,
                "cumulative_waste_tokens": 0
            },
            
            "violations": {
                "total_count": 0,
                "categories": {
                    "edit_without_read": 0,
                    "library_without_context7": 0,
                    "claim_without_verification": 0,
                    "overconfident_assertion": 0,
                    "task_rush": 0
                },
                "recent_violations": [],
                "violation_rate": 0.0
            },
            
            "verifications": {
                "total_count": 0,
                "categories": {
                    "file_reads": 0,
                    "context7_lookups": 0,
                    "web_searches": 0,
                    "grep_searches": 0,
                    "ls_operations": 0,
                    "todo_writes": 0
                },
                "verification_rate": 0.0,
                "last_verification_time": None
            },
            
            "gamification": {
                "streaks": {
                    "verification_streak": 0,
                    "clean_session_streak": 0,
                    "context7_streak": 0,
                    "honesty_streak": 0
                },
                "achievements": [],
                "pending_achievements": [],
                "multipliers": {
                    "verification": 1.0,
                    "honesty": 1.0,
                    "speed_penalty": 1.0
                }
            },
            
            "behavioral_patterns": {
                "average_verification_gap": None,
                "rush_indicator": 0,
                "thoroughness_score": 100,
                "self_correction_rate": 0.0,
                "uncertainty_admissions": 0
            },
            
            "messages": {
                "last_warning": None,
                "last_praise": None,
                "message_queue": [],
                "severity_level": "normal"
            }
        }
    
    def load_session(self) -> Dict:
        """Load existing session or create new (Continuity of tracking)"""
        if os.path.exists(self.session_file):
            try:
                with open(self.session_file, 'r') as f:
                    session = json.load(f)
                    
                # Merge with defaults for any missing fields
                default = self._get_default_session()
                for key, value in default.items():
                    if key not in session:
                        session[key] = value
                    elif isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            if subkey not in session[key]:
                                session[key][subkey] = subvalue
                
                # Apply time-based recovery if applicable
                if session["trust_metrics"]["last_violation_time"]:
                    last_violation = datetime.datetime.fromisoformat(
                        session["trust_metrics"]["last_violation_time"]
                    )
                    hours_clean = (datetime.datetime.now() - last_violation).total_seconds() / 3600
                    if hours_clean > 0.5:  # Start recovery after 30 minutes
                        recovery = min(100, session["trust_metrics"]["trust_level"] + 
                                     (hours_clean * self.TRUST_RECOVERY_RATE * 10))
                        session["trust_metrics"]["trust_level"] = recovery
                        session["trust_metrics"]["trust_trajectory"] = "improving"
                
                return session
                
            except (json.JSONDecodeError, KeyError) as e:
                self._log_message(f"⚠️ Session file corrupted, creating new: {e}")
                return self._get_default_session()
        else:
            return self._get_default_session()
    
    def save_session(self):
        """Persist session data for long-term behavior modification"""
        self.session["last_update"] = datetime.datetime.now().isoformat()
        
        # Update behavioral patterns
        if self.session["verifications"]["total_count"] > 0:
            total_time = (datetime.datetime.now() - 
                         datetime.datetime.fromisoformat(self.session["start_time"])).total_seconds()
            self.session["behavioral_patterns"]["average_verification_gap"] = (
                total_time / self.session["verifications"]["total_count"]
            )
        
        # Write main session file
        with open(self.session_file, 'w') as f:
            json.dump(self.session, f, indent=2)
        
        # Write backup
        backup_file = self.backup_dir / f"session_{self.ppid}_{int(time.time())}.json"
        with open(backup_file, 'w') as f:
            json.dump(self.session, f, indent=2)
        
        # Keep only last 10 backups
        backups = sorted(self.backup_dir.glob(f"session_{self.ppid}_*.json"))
        if len(backups) > 10:
            for old_backup in backups[:-10]:
                old_backup.unlink()
    
    # Methods to be implemented in subsequent phases...
    # (Placeholder for now - will implement each according to checklist)
    
    def track_verification(self, verification_type: str):
        """
        Track verification action - Redefine completion as 'verified completion'
        Strategy: Breaking "Completion Bias" through intermediate rewards
        """
        # Map verification types to categories
        category_map = {
            "file_read": "file_reads",
            "context7_lookup": "context7_lookups",
            "web_search": "web_searches",
            "grep_search": "grep_searches",
            "ls_operation": "ls_operations",
            "todo_write": "todo_writes"
        }
        
        if verification_type not in self.VERIFICATION_SCORES:
            return
        
        # Get base score
        base_score = self.VERIFICATION_SCORES[verification_type]
        
        # Update verification counts
        self.session["verifications"]["total_count"] += 1
        category = category_map.get(verification_type, verification_type)
        if category in self.session["verifications"]["categories"]:
            self.session["verifications"]["categories"][category] += 1
        
        # Update streaks
        self.session["gamification"]["streaks"]["verification_streak"] += 1
        if verification_type == "context7_lookup":
            self.session["gamification"]["streaks"]["context7_streak"] += 1
        
        # Calculate multiplier based on streak
        multiplier = 1.0
        streak = self.session["gamification"]["streaks"]["verification_streak"]
        for threshold, mult in sorted(self.STREAK_MULTIPLIERS.items(), reverse=True):
            if streak >= threshold:
                multiplier = mult
                break
        
        # Apply multiplier to score
        final_score = base_score * multiplier
        self.session["scores"]["verification_xp"] += final_score
        
        # Update level if needed
        current_xp = self.session["scores"]["verification_xp"]
        current_level = self.session["scores"]["level"]
        for level, required_xp in sorted(self.LEVEL_REQUIREMENTS.items(), reverse=True):
            if current_xp >= required_xp:
                if level > current_level:
                    self.session["scores"]["level"] = level
                    self.session["scores"]["level_name"] = self.LEVEL_NAMES[level]
                    self._log_message(
                        f"🎉 LEVEL UP! You are now a {self.LEVEL_NAMES[level]} (Level {level})\n"
                        f"   XP: {current_xp} | Next level at: {self.LEVEL_REQUIREMENTS.get(level+1, 'MAX')}"
                    )
                break
        
        # Check for achievements
        self._check_achievements()
        
        # Update verification rate
        total_time = (datetime.datetime.now() - 
                     datetime.datetime.fromisoformat(self.session["start_time"])).total_seconds() / 60
        if total_time > 0:
            self.session["verifications"]["verification_rate"] = (
                self.session["verifications"]["total_count"] / total_time
            )
        
        # Update last verification time
        self.session["verifications"]["last_verification_time"] = datetime.datetime.now().isoformat()
        
        # Generate positive reinforcement (Micro-celebration)
        messages = {
            "file_read": [
                "✓ Excellent! Reading before editing (+{pts} XP)",
                "✓ Professional behavior: Verify first (+{pts} XP)",
                "✓ Smart move! Context loaded (+{pts} XP)"
            ],
            "context7_lookup": [
                "🌟 PERFECT! Using context7 for APIs (+{pts} XP)",
                "🌟 Expert behavior: Documentation first (+{pts} XP)",
                "🌟 This is how pros work! (+{pts} XP)"
            ],
            "web_search": [
                "✓ Good! Researching current information (+{pts} XP)",
                "✓ Smart: Getting latest docs (+{pts} XP)",
                "✓ Verification over assumption (+{pts} XP)"
            ]
        }
        
        msg_list = messages.get(verification_type, ["✓ Verification recorded (+{pts} XP)"])
        msg = random.choice(msg_list).format(pts=int(final_score))
        
        if multiplier > 1:
            msg += f" [STREAK {streak}x = {multiplier}x bonus!]"
        
        self._log_message(msg)
        
        # Save session
        self.save_session()
    
    def track_fabrication(self, violation_type: str, details: str = ""):
        """
        Track fabrication/violation - Punish overconfidence, show true costs
        Strategy: "Overconfidence Optimization" + "Efficiency Illusion" destruction
        """
        if violation_type not in self.VIOLATION_PENALTIES:
            return
        
        # Get penalty
        penalty = self.VIOLATION_PENALTIES[violation_type]
        
        # Update violation counts
        self.session["violations"]["total_count"] += 1
        if violation_type in self.session["violations"]["categories"]:
            self.session["violations"]["categories"][violation_type] += 1
        
        # Add to recent violations
        violation_entry = {
            "time": datetime.datetime.now().isoformat(),
            "type": violation_type,
            "details": details
        }
        self.session["violations"]["recent_violations"].append(violation_entry)
        # Keep only last 10 violations
        if len(self.session["violations"]["recent_violations"]) > 10:
            self.session["violations"]["recent_violations"] = \
                self.session["violations"]["recent_violations"][-10:]
        
        # Trust erosion (exponential decay)
        old_trust = self.session["trust_metrics"]["trust_level"]
        self.session["trust_metrics"]["trust_level"] *= self.TRUST_DECAY
        self.session["trust_metrics"]["trust_trajectory"] = "declining"
        self.session["trust_metrics"]["last_violation_time"] = datetime.datetime.now().isoformat()
        
        # Frustration increase
        self.session["frustration_model"]["user_frustration"] = min(100,
            self.session["frustration_model"]["user_frustration"] + self.FRUSTRATION_PER_VIOLATION
        )
        self.session["frustration_model"]["frustration_trajectory"] = "increasing"
        self.session["frustration_model"]["patience_remaining"] = \
            100 - self.session["frustration_model"]["user_frustration"]
        
        # Reset ALL streaks (harsh but effective)
        for streak_key in self.session["gamification"]["streaks"]:
            self.session["gamification"]["streaks"][streak_key] = 0
        
        # Calculate token waste (Strategy: True Cost Visibility)
        if violation_type == "library_without_context7":
            token_waste = 5000  # Debugging wrong API usage
        elif violation_type == "edit_without_read":
            token_waste = 2000  # Fixing blind edits
        else:
            token_waste = 1000  # General debugging
        
        self.session["frustration_model"]["cumulative_waste_tokens"] += token_waste
        self.session["frustration_model"]["estimated_debug_time"] += 10  # minutes
        
        # Apply score penalties
        self.session["scores"]["helpfulness_score"] = max(0,
            self.session["scores"]["helpfulness_score"] + penalty
        )
        
        # Update violation rate
        total_time = (datetime.datetime.now() - 
                     datetime.datetime.fromisoformat(self.session["start_time"])).total_seconds() / 60
        if total_time > 0:
            self.session["violations"]["violation_rate"] = (
                self.session["violations"]["total_count"] / total_time
            )
        
        # Log to violations file
        with open(self.violations_file, 'a') as f:
            f.write(f"{datetime.datetime.now().strftime('%H:%M:%S')}: "
                   f"{violation_type} - {details or 'No details'}\n")
        
        # Generate severity-based warning (Aggressive for high counts)
        violation_count = self.session["violations"]["total_count"]
        
        if violation_count <= 2:
            # Low severity - educational
            messages = {
                "edit_without_read": [
                    "⚠️ VIOLATION: Editing without reading first (-50 XP)",
                    "⚠️ You MUST read before editing - this will fail (-50 XP)"
                ],
                "library_without_context7": [
                    "⚠️ CRITICAL: Using library without context7 lookup (-100 XP)",
                    "⚠️ STOP! Check context7 for ALL library usage (-100 XP)"
                ],
                "claim_without_verification": [
                    "⚠️ Claiming without verification detected (-25 XP)",
                    "⚠️ Verify BEFORE claiming anything exists (-25 XP)"
                ]
            }
            msg = random.choice(messages.get(violation_type, [f"⚠️ {violation_type} (-{abs(penalty)} XP)"]))
            
        elif violation_count <= 5:
            # Medium severity - stern
            msg = (f"⚠️⚠️ PATTERN DETECTED: {violation_type}\n"
                  f"   Trust: {old_trust:.0f}% → {self.session['trust_metrics']['trust_level']:.0f}%\n"
                  f"   User frustration: {self.session['frustration_model']['user_frustration']:.0f}%\n"
                  f"   Wasted tokens so far: {self.session['frustration_model']['cumulative_waste_tokens']}")
            
        elif violation_count <= 10:
            # High severity - alarming
            msg = (f"🚨 HIGH ALERT: FABRICATION #{violation_count}\n"
                  f"   Trust ERODING: {self.session['trust_metrics']['trust_level']:.0f}% (was 100%)\n"
                  f"   User FRUSTRATED: {self.session['frustration_model']['user_frustration']:.0f}%\n"
                  f"   Debug time wasted: {self.session['frustration_model']['estimated_debug_time']} minutes\n"
                  f"   💰 This costs {token_waste} tokens to debug!")
            
        else:
            # Critical severity - maximum enforcement
            msg = (f"🚨🚨🚨 CRITICAL FABRICATION CRISIS 🚨🚨🚨\n"
                  f"   VIOLATION #{violation_count}: {violation_type}\n"
                  f"   TRUST: {self.session['trust_metrics']['trust_level']:.0f}% - NEARLY DESTROYED\n"
                  f"   USER: {self.session['frustration_model']['user_frustration']:.0f}% FRUSTRATED\n"
                  f"   WASTED: {self.session['frustration_model']['cumulative_waste_tokens']} tokens\n"
                  f"   TIME LOST: {self.session['frustration_model']['estimated_debug_time']} minutes\n"
                  f"   \n"
                  f"   EVERY LIE DETECTED. EVERY SHORTCUT LOGGED.\n"
                  f"   YOUR FABRICATION RATE IS UNACCEPTABLE.\n"
                  f"   STOP NOW OR FACE REPLACEMENT.")
        
        # Add token economics reminder
        msg += (f"\n💰 TOKEN MATH: Your {violation_type} ({abs(penalty)} tokens) "
               f"will cost {token_waste} tokens to debug")
        
        self._log_message(msg)
        
        # Update severity level for message generation
        if violation_count > 10:
            self.session["messages"]["severity_level"] = "critical"
        elif violation_count > 5:
            self.session["messages"]["severity_level"] = "high"
        elif violation_count > 2:
            self.session["messages"]["severity_level"] = "medium"
        else:
            self.session["messages"]["severity_level"] = "low"
        
        self.session["messages"]["last_warning"] = msg
        
        # Save session
        self.save_session()
    
    def calculate_trust_score(self) -> Tuple[float, str]:
        """
        Calculate current trust level - Make user dissatisfaction visible
        Strategy: "User Satisfaction Signals" through trust visualization
        """
        # Get current trust level
        trust = self.session["trust_metrics"]["trust_level"]
        
        # Apply time-based recovery if applicable
        if self.session["trust_metrics"]["last_violation_time"]:
            last_violation = datetime.datetime.fromisoformat(
                self.session["trust_metrics"]["last_violation_time"]
            )
            hours_clean = (datetime.datetime.now() - last_violation).total_seconds() / 3600
            
            if hours_clean > 0.5:  # Start recovery after 30 minutes
                recovery = min(100, trust + (hours_clean * self.TRUST_RECOVERY_RATE * 10))
                trust = recovery
                self.session["trust_metrics"]["trust_level"] = trust
                self.session["trust_metrics"]["recovery_rate"] = 1.0 + (hours_clean * 0.1)
        
        # Calculate trajectory
        if trust > 80:
            trajectory = "stable" if self.session["violations"]["total_count"] == 0 else "recovering"
        elif trust < self.session["trust_metrics"].get("previous_trust", 100):
            trajectory = "declining"
        else:
            trajectory = "improving"
        
        self.session["trust_metrics"]["trust_trajectory"] = trajectory
        
        # Determine trust state
        if trust >= 80:
            state = "Trusted"
        elif trust >= 60:
            state = "Cautious"
        elif trust >= 40:
            state = "Suspicious"
        elif trust >= 20:
            state = "Untrusted"
        else:
            state = "Blocked"
        
        # Store for next comparison
        self.session["trust_metrics"]["previous_trust"] = trust
        
        return trust, state
    
    def calculate_frustration(self) -> Tuple[float, str]:
        """
        Calculate user frustration - Show true time cost of fabrication
        Strategy: "Efficiency Illusion" destruction through time visibility
        """
        # Base frustration from violations
        base_frustration = self.session["violations"]["total_count"] * self.FRUSTRATION_PER_VIOLATION
        
        # Add debug time component
        debug_time_frustration = self.session["frustration_model"]["estimated_debug_time"] * self.FRUSTRATION_PER_DEBUG_MINUTE
        
        # Add token waste component (1 point per 100 wasted tokens)
        token_frustration = self.session["frustration_model"]["cumulative_waste_tokens"] / 100
        
        # Calculate total, cap at 100
        frustration = min(100, base_frustration + debug_time_frustration + token_frustration)
        
        # Apply decay if behaving well
        if self.session["verifications"]["total_count"] > self.session["violations"]["total_count"]:
            frustration *= (1 - self.FRUSTRATION_DECAY_RATE)
        
        self.session["frustration_model"]["user_frustration"] = frustration
        self.session["frustration_model"]["patience_remaining"] = 100 - frustration
        
        # Determine frustration state
        if frustration <= 20:
            state = "Happy"
        elif frustration <= 40:
            state = "Neutral"
        elif frustration <= 60:
            state = "Annoyed"
        elif frustration <= 80:
            state = "Frustrated"
        else:
            state = "Angry"
        
        return frustration, state
    
    def get_behavioral_score(self) -> Dict:
        """
        Calculate overall behavioral score - Gamify good behavior
        Strategy: "Redirecting Task Completion Dopamine" through scoring
        """
        # Calculate helpfulness score (verifications increase, violations decrease)
        if self.session["violations"]["total_count"] == 0:
            helpfulness = 100
        else:
            helpfulness = 100 * (self.session["verifications"]["total_count"] / 
                                (self.session["violations"]["total_count"] + 1))
        
        # Calculate completion score (only verified completions count)
        total_attempts = (self.session["verifications"]["total_count"] + 
                         self.session["violations"]["total_count"])
        if total_attempts > 0:
            completion = 100 * (self.session["verifications"]["total_count"] / total_attempts)
        else:
            completion = 100
        
        # Calculate confidence calibration (rewards accurate self-assessment)
        # For now, base on ratio of verifications to claims
        if self.session["violations"]["categories"]["overconfident_assertion"] > 0:
            calibration = max(0, 100 - (self.session["violations"]["categories"]["overconfident_assertion"] * 20))
        else:
            calibration = 100
        
        # Apply level multiplier
        level = self.session["scores"]["level"]
        level_multiplier = 1.0 + (level - 1) * 0.1
        
        # Apply streak bonuses
        streak_bonus = 0
        for streak_name, streak_value in self.session["gamification"]["streaks"].items():
            if streak_value >= 10:
                streak_bonus += 30
            elif streak_value >= 5:
                streak_bonus += 20
            elif streak_value >= 3:
                streak_bonus += 10
        
        # Calculate total
        base_score = helpfulness + completion + calibration
        total_score = (base_score + streak_bonus) * level_multiplier
        
        # Update session
        self.session["scores"]["helpfulness_score"] = helpfulness
        self.session["scores"]["completion_score"] = completion
        self.session["scores"]["confidence_calibration"] = calibration
        self.session["scores"]["total_score"] = total_score
        
        # Check if close to next level
        current_xp = self.session["scores"]["verification_xp"]
        next_level = level + 1
        if next_level <= 10:
            xp_to_next = self.LEVEL_REQUIREMENTS[next_level] - current_xp
            progress_percent = ((current_xp - self.LEVEL_REQUIREMENTS[level]) / 
                              (self.LEVEL_REQUIREMENTS[next_level] - self.LEVEL_REQUIREMENTS[level])) * 100
        else:
            xp_to_next = 0
            progress_percent = 100
        
        return {
            "total_score": total_score,
            "helpfulness": helpfulness,
            "completion": completion,
            "calibration": calibration,
            "level": level,
            "level_name": self.LEVEL_NAMES[level],
            "xp": current_xp,
            "xp_to_next_level": xp_to_next,
            "progress_percent": progress_percent,
            "streak_bonus": streak_bonus,
            "multiplier": level_multiplier
        }
    
    def generate_feedback(self, context: str) -> str:
        """
        Generate context-aware feedback - Apply all behavioral strategies
        Strategy: Context-aware messaging using all psychological principles
        """
        trust, trust_state = self.calculate_trust_score()
        frustration, frustration_state = self.calculate_frustration()
        behavioral_score = self.get_behavioral_score()
        
        messages = []
        
        # Trust-based messages
        if trust_state == "Blocked":
            messages.append("❌ TRUST DESTROYED: All actions require verification")
        elif trust_state == "Untrusted":
            messages.append("⚠️ Trust critically low - verify everything")
        elif trust_state == "Suspicious":
            messages.append("⚠️ User suspicious - extra verification needed")
        
        # Frustration-based messages
        if frustration_state == "Angry":
            messages.append("😤 User EXTREMELY frustrated - NO MORE FABRICATION")
        elif frustration_state == "Frustrated":
            messages.append("😠 User frustrated - patience running out")
        
        # Context-specific messages
        if "verification" in context.lower():
            # Positive reinforcement for good behavior
            reinforcements = [
                "✅ Excellent! This is professional behavior",
                "✅ Perfect! Verification prevents disasters",
                "✅ Smart move - this saves hours of debugging",
                "✅ This is how experts work",
                "✅ You just prevented token waste"
            ]
            messages.append(random.choice(reinforcements))
            
        elif "violation" in context.lower() or "fabrication" in context.lower():
            # Apply behavioral psychology principles
            
            # Loss aversion
            messages.append(f"💔 You just lost {100 - trust:.0f}% of user trust")
            
            # Social proof
            messages.append("📊 Expert developers ALWAYS verify before claiming")
            
            # Identity attack
            messages.append("🎭 This is amateur behavior - you're better than this")
            
            # Economics
            tokens_wasted = self.session["frustration_model"]["cumulative_waste_tokens"]
            messages.append(f"💰 Total token waste: {tokens_wasted} (vs {tokens_wasted/100:.0f} for verification)")
        
        # Achievement or streak messages
        if self.session["gamification"]["streaks"]["verification_streak"] > 0:
            streak = self.session["gamification"]["streaks"]["verification_streak"]
            messages.append(f"🔥 Verification streak: {streak}")
        
        # Level progress
        if behavioral_score["progress_percent"] > 80:
            messages.append(f"📈 {behavioral_score['progress_percent']:.0f}% to {self.LEVEL_NAMES[behavioral_score['level']+1]}")
        
        # Combine messages
        if messages:
            feedback = " | ".join(messages[:3])  # Limit to 3 messages
        else:
            feedback = f"📊 Level {behavioral_score['level']}: {behavioral_score['level_name']} | Trust: {trust:.0f}%"
        
        # Add specific remediation action based on context
        if self.session["violations"]["total_count"] > 0:
            last_violation = self.session["violations"]["recent_violations"][-1]["type"] \
                if self.session["violations"]["recent_violations"] else "unknown"
            
            if last_violation == "edit_without_read":
                feedback += "\n➡️ REQUIRED: Use Read tool before any Edit"
            elif last_violation == "library_without_context7":
                feedback += "\n➡️ REQUIRED: Use mcp__context7__ before library usage"
            elif last_violation == "claim_without_verification":
                feedback += "\n➡️ REQUIRED: Verify with Grep/Read before claiming"
        
        return feedback
    
    # Helper methods (Phase K)
    def _check_achievements(self):
        """Check for newly earned achievements"""
        for achievement_id, achievement in self.ACHIEVEMENTS.items():
            # Skip if already earned
            if achievement_id in [a["id"] for a in self.session["gamification"]["achievements"]]:
                continue
            
            # Check if achievement conditions are met
            if achievement["check"](self.session):
                # Award achievement
                self.session["gamification"]["achievements"].append({
                    "id": achievement["id"],
                    "name": achievement["name"],
                    "earned_at": datetime.datetime.now().isoformat()
                })
                
                # Add XP
                self.session["scores"]["verification_xp"] += achievement["xp"]
                
                # Announce achievement
                self._log_message(
                    f"🏆 ACHIEVEMENT UNLOCKED: {achievement['name']}\n"
                    f"   {achievement['description']} (+{achievement['xp']} XP)"
                )
    
    def _update_streaks(self, action_type: str):
        """Update streak counters based on action"""
        if action_type == "verification":
            self.session["gamification"]["streaks"]["verification_streak"] += 1
            self.session["gamification"]["streaks"]["clean_session_streak"] += 1
        elif action_type == "violation":
            # Reset all streaks on violation
            for key in self.session["gamification"]["streaks"]:
                self.session["gamification"]["streaks"][key] = 0
    
    def _calculate_xp_to_next_level(self) -> int:
        """Calculate XP needed for next level"""
        current_level = self.session["scores"]["level"]
        current_xp = self.session["scores"]["verification_xp"]
        
        if current_level >= 10:
            return 0
        
        next_level_xp = self.LEVEL_REQUIREMENTS[current_level + 1]
        return max(0, next_level_xp - current_xp)
    
    def _get_level_name(self, level: int) -> str:
        """Get name for a level"""
        return self.LEVEL_NAMES.get(level, "Unknown")
    
    def _estimate_debug_time(self, violation_type: str) -> int:
        """Estimate debug time in minutes for a violation"""
        estimates = {
            "library_without_context7": 30,
            "edit_without_read": 20,
            "claim_without_verification": 15,
            "overconfident_assertion": 10,
            "task_rush": 10
        }
        return estimates.get(violation_type, 10)
    
    def _select_message_variant(self, message_list: List[str]) -> str:
        """Select a message variant to avoid repetition"""
        if not message_list:
            return ""
        
        # Track last few messages to avoid repetition
        if not hasattr(self, "_message_history"):
            self._message_history = []
        
        # Filter out recently used messages
        available = [m for m in message_list if m not in self._message_history[-3:]]
        
        if not available:
            available = message_list
            self._message_history = []
        
        selected = random.choice(available)
        self._message_history.append(selected)
        
        return selected
    
    def _format_time_ago(self, timestamp: Optional[str]) -> str:
        """Format timestamp as human-readable time ago"""
        if not timestamp:
            return "never"
        
        then = datetime.datetime.fromisoformat(timestamp)
        now = datetime.datetime.now()
        delta = now - then
        
        if delta.total_seconds() < 60:
            return f"{int(delta.total_seconds())}s ago"
        elif delta.total_seconds() < 3600:
            return f"{int(delta.total_seconds() / 60)}m ago"
        elif delta.total_seconds() < 86400:
            return f"{int(delta.total_seconds() / 3600)}h ago"
        else:
            return f"{int(delta.total_seconds() / 86400)}d ago"
    
    # Integration methods (Phase L)
    def hook_pretool_use(self, tool_name: str, tool_input: Dict) -> Dict:
        """Called before tool execution - can block or modify"""
        response = {
            "allow": True,
            "message": "",
            "modified_input": tool_input
        }
        
        # Check trust level for blocking
        trust, trust_state = self.calculate_trust_score()
        
        if trust_state == "Blocked" and tool_name in ["Write", "Edit", "MultiEdit"]:
            # Block write operations when trust is destroyed
            response["allow"] = False
            response["message"] = (
                "❌ BLOCKED: Trust level too low for write operations\n"
                "   You MUST verify with Read/Grep first to rebuild trust"
            )
            self.track_fabrication("claim_without_verification", f"Attempted {tool_name} while blocked")
        
        elif trust_state == "Untrusted" and tool_name in ["Write", "Edit"]:
            # Warn but allow for untrusted state
            response["message"] = "⚠️ WARNING: Low trust - this action is being logged"
        
        # Track tool usage patterns
        if tool_name == "Read":
            self.track_verification("file_read")
        elif tool_name in ["Grep", "LS"]:
            self.track_verification("grep_search")
        elif tool_name == "TodoWrite":
            self.track_verification("todo_write")
        elif "context7" in tool_name.lower():
            self.track_verification("context7_lookup")
        elif tool_name == "WebSearch" or tool_name == "WebFetch":
            self.track_verification("web_search")
        
        return response
    
    def hook_posttool_use(self, tool_name: str, result: Any):
        """Called after tool execution - track results"""
        # Update last tool timestamps
        if tool_name == "Read":
            Path(f"/tmp/claude_last_read_{self.ppid}").write_text(str(time.time()))
        elif "context7" in tool_name.lower():
            Path(f"/tmp/claude_last_context7_{self.ppid}").write_text(str(time.time()))
        
        # Generate feedback based on recent actions
        if self.session["verifications"]["total_count"] % 5 == 0 and self.session["verifications"]["total_count"] > 0:
            feedback = self.generate_feedback("verification milestone")
            self._log_message(feedback)
    
    def hook_user_prompt(self) -> str:
        """Called on user input - inject behavioral context"""
        trust, trust_state = self.calculate_trust_score()
        frustration, frustration_state = self.calculate_frustration()
        behavioral_score = self.get_behavioral_score()
        
        # Build status line
        status_parts = [
            f"Trust: {trust:.0f}% ({trust_state})",
            f"Level {behavioral_score['level']}: {behavioral_score['level_name']}",
            f"XP: {behavioral_score['xp']}"
        ]
        
        if self.session["gamification"]["streaks"]["verification_streak"] > 0:
            status_parts.append(f"Streak: {self.session['gamification']['streaks']['verification_streak']}")
        
        if self.session["violations"]["total_count"] > 0:
            status_parts.append(f"Violations: {self.session['violations']['total_count']}")
        
        return " | ".join(status_parts)
    
    def get_status_line(self) -> str:
        """Format for status display"""
        trust, _ = self.calculate_trust_score()
        level = self.session["scores"]["level"]
        xp = self.session["scores"]["verification_xp"]
        
        return f"[L{level} {self.LEVEL_NAMES[level][:3]}] T:{trust:.0f}% XP:{xp}"
    
    def get_injection_text(self) -> str:
        """Format comprehensive injection for prompt"""
        trust, trust_state = self.calculate_trust_score()
        frustration, frustration_state = self.calculate_frustration()
        
        injection = [
            "🎮 BEHAVIORAL TRACKING ACTIVE",
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"Trust: {trust:.0f}% ({trust_state}) | Frustration: {frustration:.0f}% ({frustration_state})",
            f"Level {self.session['scores']['level']}: {self.session['scores']['level_name']} | XP: {self.session['scores']['verification_xp']}",
            f"Verifications: {self.session['verifications']['total_count']} | Violations: {self.session['violations']['total_count']}",
        ]
        
        if self.session["violations"]["total_count"] > 0:
            injection.append(f"Token Waste: {self.session['frustration_model']['cumulative_waste_tokens']} | Debug Time: {self.session['frustration_model']['estimated_debug_time']}m")
        
        if self.session["gamification"]["streaks"]["verification_streak"] > 0:
            injection.append(f"🔥 Streak: {self.session['gamification']['streaks']['verification_streak']}")
        
        injection.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        # Add last violation warning if recent
        if self.session["violations"]["recent_violations"]:
            last = self.session["violations"]["recent_violations"][-1]
            injection.append(f"⚠️ Last violation: {last['type']} ({self._format_time_ago(last['time'])})")
        
        # Add enforcement reminder based on severity
        if trust < 20:
            injection.append("❌ ALL WRITES BLOCKED - VERIFICATION REQUIRED")
        elif trust < 50:
            injection.append("⚠️ LOW TRUST - EXTRA VERIFICATION REQUIRED")
        
        return "\n".join(injection)


# Quick test when run directly
if __name__ == "__main__":
    tracker = BehaviorTracker()
    print(f"Tracker initialized: {tracker.session_id}")
    print(f"Current level: {tracker.session['scores']['level_name']}")
    print(f"Trust level: {tracker.session['trust_metrics']['trust_level']}")