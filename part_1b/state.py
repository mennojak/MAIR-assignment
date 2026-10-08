from typing import Optional


REQUIRED_SLOTS = ("food", "area", "pricerange")
ADDITIONAL_REQUIREMENTS = ("touristic", "assigned_seats", "children", "romantic")


class DialogueState:
    def __init__(self):
        self.current_state = "1_welcome"
        self.requirements = {slot: None for slot in REQUIRED_SLOTS}
        self.additional_requirements = {
            slot: None for slot in ADDITIONAL_REQUIREMENTS
        } 
        self.additional_requirements_asked = False
        self.matches = []
        self.previous_recommendations = []
        self.current_recommendation = None
        self.pending_confirmation = None
        self.last_utterance = ""
        self.dialog_act = None
        self.reasoning = None

    def update_from_user_input(
        self,
        utterance: str,
        dialog_act: str,
        extraction: Optional[dict],
    ):
        self.last_utterance = utterance
        self.dialog_act = dialog_act

        if self.current_state == "7_ask_additional_requirements":
            self.additional_requirements_asked = True

        if self.pending_confirmation is not None:
            if dialog_act.lower() == "affirm":
                slot = self.pending_confirmation["slot"]
                self.requirements[slot] = self.pending_confirmation["suggestion"]
                self.pending_confirmation = None
            elif dialog_act.lower() == "negate":
                self.pending_confirmation = None
            return

        if extraction is None:
            return

        for name, value in extraction["slots"].items():
            if name in REQUIRED_SLOTS:
                self.requirements[name] = value
            elif name in ADDITIONAL_REQUIREMENTS:
                self.additional_requirements[name] = value

        confirmation = extraction["confirmation"]
        if confirmation is not None:
            slot = confirmation["slot"]

            self.pending_confirmation = {
                "slot": slot,
                "original": confirmation["original"],
                "suggestion": confirmation["suggestion"],
            }

    def transition(self, dialog_act: str, requirements_changed: bool = False):
        dialog_act = dialog_act.lower()

        if dialog_act in ("bye", "thankyou"):
            self.current_state = "10_goodbye"
            return

        if dialog_act == "restart":
            self.__init__()
            return

        if self.pending_confirmation is not None:
            self.current_state = "5_confirm_value"
            return

        if dialog_act == "request" and self.current_recommendation is not None:
            self.current_state = "9_give_info"
            return

        if self.current_state == "9_give_info":
            self.current_state = "8_suggest_restaurant"

        if dialog_act == "reqalts" and not requirements_changed:
            if self.matches:
                self.current_state = "8_suggest_restaurant"
            else:
                self.current_state = "6_no_match"
            return

        if self.requirements["food"] is None:
            self.current_state = "2_ask_food"
            return
        if self.requirements["area"] is None:
            self.current_state = "3_ask_area"
            return
        if self.requirements["pricerange"] is None:
            self.current_state = "4_ask_price"
            return

        if not self.matches:
            self.current_state = "6_no_match"
            return

        if (
            len(self.matches) > 1
            and not self.additional_requirements_asked
            and self.current_state != "6_no_match"
        ):
            self.current_state = "7_ask_additional_requirements"
            return

        self.current_state = "8_suggest_restaurant"