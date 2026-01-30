"""
HTTP Client for Affordance Nodes

Provides a wrapper around httpx for communicating with Thing Description endpoints.
Supports synchronous operations with retry logic, timeouts, and error handling.
"""

import httpx
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class HTTPError(Exception):
    """
    Exception raised when an HTTP request fails.
    
    Attributes:
        url: The URL that was requested
        status_code: HTTP status code (if available)
        message: Error description
        response_body: Response body content (if available)
    """
    
    def __init__(
        self,
        url: str,
        status_code: Optional[int] = None,
        message: str = "HTTP request failed",
        response_body: Optional[str] = None
    ):
        self.url = url
        self.status_code = status_code
        self.message = message
        self.response_body = response_body
        super().__init__(f"{message} (URL: {url}, Status: {status_code})")


@dataclass
class HTTPClientConfig:
    """
    Configuration for HTTP client behavior.
    
    Attributes:
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts for failed requests
        retry_on_status_codes: HTTP status codes that trigger a retry
        default_headers: Headers to include in all requests
        verify_ssl: Whether to verify SSL certificates (for HTTPS)
    """
    
    timeout: float = 30.0
    max_retries: int = 3
    retry_on_status_codes: Tuple[int, ...] = (408, 429, 500, 502, 503, 504)
    default_headers: Dict[str, str] = field(default_factory=lambda: {
        "Accept": "application/json",
        "Content-Type": "application/json"
    })
    verify_ssl: bool = True
    
    def __post_init__(self):
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")


@dataclass
class HTTPResponse:
    """
    Encapsulates an HTTP response.
    
    Attributes:
        status_code: HTTP status code
        body: Response body (parsed as JSON if possible, otherwise raw string)
        headers: Response headers as a dictionary
        url: The final URL (after any redirects)
        elapsed_time: Time taken for the request in seconds
    """
    
    status_code: int
    body: Any
    headers: Dict[str, str]
    url: str
    elapsed_time: float
    
    @property
    def is_success(self) -> bool:
        """Check if the response indicates success (2xx status code)."""
        return 200 <= self.status_code < 300
    
    @property
    def is_json(self) -> bool:
        """Check if the response body is a parsed JSON object."""
        return isinstance(self.body, (dict, list))


class HTTPClient:
    """
    HTTP client wrapper around httpx for interacting with Thing Description endpoints.
    
    This client provides methods for making HTTP requests to action and property
    affordance endpoints, with built-in retry logic, timeout handling, and
    consistent error reporting.
    
    Expected response formats from simulators:
    - Property reads (GET): JSON values (e.g., "on", 25, {"hand": "empty", "blocks": [...]})
    - Action invocations (POST): {"status": "success", "message": "..."} on success
    - Errors: {"error": "..."} or {"detail": "..."} with HTTP 4xx/5xx
    
    Example:
        client = HTTPClient()
        
        # Invoke an action
        response = client.post(
            "http://localhost:8080/artifacts/light/turn_on",
            payload={}
        )
        
        # Read a property
        response = client.get(
            "http://localhost:8080/artifacts/light/properties/state"
        )
        print(response.body)  # "on" or {"hand": "empty", ...}
    """
    
    def __init__(self, config: Optional[HTTPClientConfig] = None):
        """
        Initialize the HTTP client.
        
        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or HTTPClientConfig()
        self._session_headers: Dict[str, str] = {}
        
        # Create httpx transport with retries
        transport = httpx.HTTPTransport(retries=self.config.max_retries)
        
        # Create httpx client
        self._client = httpx.Client(
            timeout=httpx.Timeout(self.config.timeout),
            headers=self.config.default_headers,
            transport=transport,
            verify=self.config.verify_ssl,
        )
    
    def set_session_header(self, key: str, value: str) -> None:
        """
        Set a header that will be included in all requests for this session.
        
        Args:
            key: Header name
            value: Header value
        """
        self._session_headers[key] = value
        self._client.headers[key] = value
    
    def clear_session_headers(self) -> None:
        """Clear all session-specific headers."""
        for key in self._session_headers:
            self._client.headers.pop(key, None)
        self._session_headers.clear()
    
    def _convert_response(self, response: httpx.Response) -> HTTPResponse:
        """Convert httpx response to our HTTPResponse format."""
        # Try to parse JSON, fall back to text
        try:
            body = response.json()
        except Exception:
            body = response.text if response.text else None
        
        return HTTPResponse(
            status_code=response.status_code,
            body=body,
            headers=dict(response.headers),
            url=str(response.url),
            elapsed_time=response.elapsed.total_seconds(),
        )
    
    def _handle_error(self, e: Exception, url: str) -> None:
        """Convert httpx exceptions to HTTPError."""
        if isinstance(e, httpx.HTTPStatusError):
            # Try to get error details from response body
            try:
                error_body = e.response.json()
                error_msg = error_body.get("error") or error_body.get("detail") or str(error_body)
            except Exception:
                error_msg = e.response.text or str(e)
            
            raise HTTPError(
                url=url,
                status_code=e.response.status_code,
                message=f"HTTP {e.response.status_code}: {error_msg}",
                response_body=e.response.text,
            )
        elif isinstance(e, httpx.TimeoutException):
            raise HTTPError(
                url=url,
                status_code=None,
                message=f"Request timed out after {self.config.timeout}s",
            )
        elif isinstance(e, httpx.ConnectError):
            raise HTTPError(
                url=url,
                status_code=None,
                message=f"Connection failed: {str(e)}",
            )
        else:
            raise HTTPError(
                url=url,
                status_code=None,
                message=f"Request failed: {str(e)}",
            )
    
    def get(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None
    ) -> HTTPResponse:
        """
        Make a GET request (used for reading property affordances).
        
        Args:
            url: The URL to request
            headers: Additional headers to include
            
        Returns:
            HTTPResponse object with body containing the property value
        """
        try:
            response = self._client.get(url, headers=headers)
            response.raise_for_status()
            return self._convert_response(response)
        except Exception as e:
            self._handle_error(e, url)
    
    def post(
        self,
        url: str,
        payload: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> HTTPResponse:
        """
        Make a POST request (used for invoking action affordances).
        
        Args:
            url: The URL to request
            payload: Request body (will be JSON-encoded)
            headers: Additional headers to include
            
        Returns:
            HTTPResponse object with body containing {"status": "success", "message": "..."}
        """
        try:
            response = self._client.post(url, json=payload or {}, headers=headers)
            response.raise_for_status()
            return self._convert_response(response)
        except Exception as e:
            self._handle_error(e, url)
    
    def put(
        self,
        url: str,
        payload: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> HTTPResponse:
        """
        Make a PUT request.
        
        Args:
            url: The URL to request
            payload: Request body (will be JSON-encoded)
            headers: Additional headers to include
            
        Returns:
            HTTPResponse object
        """
        try:
            response = self._client.put(url, json=payload or {}, headers=headers)
            response.raise_for_status()
            return self._convert_response(response)
        except Exception as e:
            self._handle_error(e, url)
    
    def delete(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None
    ) -> HTTPResponse:
        """
        Make a DELETE request.
        
        Args:
            url: The URL to request
            headers: Additional headers to include
            
        Returns:
            HTTPResponse object
        """
        try:
            response = self._client.delete(url, headers=headers)
            response.raise_for_status()
            return self._convert_response(response)
        except Exception as e:
            self._handle_error(e, url)
    
    def close(self) -> None:
        """Close the underlying httpx client."""
        self._client.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close the client."""
        self.close()
        return False
