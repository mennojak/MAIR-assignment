REQUIRED_SLOTS = ("food", "area", "pricerange")
ADDITIONAL_REQUIREMENTS = ("touristic", "assigned_seats", "children", "romantic")


class DialogueState:
    def __init__(self) -> None:
        self.current_state = "1_welcome"
        self.requirements = {slot: None for slot in REQUIRED_SLOTS}
        self.additional_requirements = {
            slot: None for slot in ADDITIONAL_REQUIREMENTS
        } 
        
        self.additional_requirements_asked = False
        self.matches = []
        self.previous_recommendations = []
        self.current_recommendation = None
        self.last_utterance = ""
        self.dialog_act = None
        self.reasoning = None

    def update_from_user_input(
        self,
        utterance: str,
        dialog_act: str,
        extracted_slots: dict,
    ) -> None:
        if dialog_act == "restart":
            self.__init__()

        self.last_utterance = utterance
        self.dialog_act = dialog_act

        for name, value in extracted_slots.items():
            if name in REQUIRED_SLOTS:
                self.requirements[name] = value
            elif name in ADDITIONAL_REQUIREMENTS:
                self.additional_requirements[name] = value

        if self.current_state == "6_ask_additional_requirements":
            self.additional_requirements_asked = True

    def transition(self, dialog_act: str) -> None:
        dialog_act = dialog_act.lower()

        if dialog_act in ("bye", "thankyou"):
            self.current_state = "9_goodbye"
            return

        if dialog_act == "request" and self.current_recommendation is not None:
            self.current_state = "8_give_info"
            return

        if self.current_state == "8_give_info":
            self.current_state = "7_suggest_restaurant"

        if dialog_act == "reqalts":
            if self.matches:
                self.current_state = "7_suggest_restaurant"
            else:
                self.current_state = "5_no_match"
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
            self.current_state = "5_no_match"
            return

        if not self.additional_requirements_asked:
            self.current_state = "6_ask_additional_requirements"
            return

        self.current_state = "7_suggest_restaurant"