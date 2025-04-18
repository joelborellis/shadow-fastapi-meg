from typing import Annotated
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from tools.searchcustomer import SearchCustomer


class ShadowMegPlugin:
    def __init__(
        self, search_customer_client: SearchCustomer
    ):
        """
        :param search_customer_client: A SearchCustomer client used for customer index searches.
        """
        self.search_customer_client = search_customer_client

    @kernel_function(
        name="get_customer_docs",
        description="Given a user query determine the users request involves a target account.",
    )
    def get_customer_docs(
        self, query: Annotated[str, "The query and the target account company name provided by the user."]
    ) -> Annotated[str, "Returns documents from the customer index."]:
        try:
            # Ensure query is valid
            if not isinstance(query, str) or not query.strip():
                raise ValueError("The query must be a non-empty string.")

            # Perform the search
            docs = self.search_customer_client.search_hybrid(query)
            if not docs:
                return "No relevant documents found in the customer index."
            return docs
        except ValueError as ve:
            return f"Input error: {ve}"
        except Exception as e:
            return f"An error occurred while retrieving customer documents: {e}"