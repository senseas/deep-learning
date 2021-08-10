
package com.deep.framework.jit.lexer;


import java.util.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.Collectors;

import static com.deep.framework.jit.lexer.TokenKind.*;

public enum TokenType {
    //@formatter:off

    // Keywords
    ABSTRACT      (KEYWORD,"abstract"),
    ASSERT        (KEYWORD,"assert"),
    BOOLEAN       (KEYWORD,"boolean"),
    BREAK         (KEYWORD,"break"),
    BYTE          (KEYWORD,"byte"),
    CASE          (KEYWORD,"case"),
    CATCH         (KEYWORD,"catch"),
    CHAR          (KEYWORD,"char"),
    CLASS         (KEYWORD,"class"),
    CONST         (KEYWORD,"const"),
    CONTINUE      (KEYWORD,"continue"),
    DEFAULT       (KEYWORD,"default"),
    DO            (KEYWORD,"do"),
    DOUBLE        (KEYWORD,"double"),
    ELSE          (KEYWORD,"else"),
    ENUM          (KEYWORD,"enum"),
    EXTENDS       (KEYWORD,"extends"),
    FINAL         (KEYWORD,"final"),
    FINALLY       (KEYWORD,"finally"),
    FLOAT         (KEYWORD,"float"),
    FOR           (KEYWORD,"for"),
    IF            (KEYWORD,"if"),
    GOTO          (KEYWORD,"goto"),
    IMPLEMENTS    (KEYWORD,"implements"),
    IMPORT        (KEYWORD,"import"),
    INSTANCEOF    (KEYWORD,"instanceof"),
    INT           (KEYWORD,"int"),
    INTERFACE     (KEYWORD,"interface"),
    LONG          (KEYWORD,"long"),
    NATIVE        (KEYWORD,"native"),
    NEW           (KEYWORD,"new"),
    PACKAGE       (KEYWORD,"package"),
    PRIVATE       (KEYWORD,"private"),
    PROTECTED     (KEYWORD,"protected"),
    PUBLIC        (KEYWORD,"public"),
    RETURN        (KEYWORD,"return"),
    SHORT         (KEYWORD,"short"),
    STATIC        (KEYWORD,"static"),
    STRICTFP      (KEYWORD,"strictfp"),
    SUPER         (KEYWORD,"super"),
    SWITCH        (KEYWORD,"switch"),
    SYNCHRONIZED  (KEYWORD,"synchronized"),
    THIS          (KEYWORD,"this"),
    THROW         (KEYWORD,"throw"),
    THROWS        (KEYWORD,"throws"),
    TRANSIENT     (KEYWORD,"transient"),
    TRY           (KEYWORD,"try"),
    VOID          (KEYWORD,"void"),
    VOLATILE      (KEYWORD,"volatile"),
    WHILE         (KEYWORD,"while"),
    UNDER_SCORE   (KEYWORD,"_"),

    // Separators
    LPAREN        (BINARY,"("),
    RPAREN        (BINARY,")"),
    LBRACE        (BINARY,"{"),
    RBRACE        (BINARY,"}"),
    LBRACK        (BINARY,"["),
    RBRACK        (BINARY,"]"),
    SEMI          (BINARY,";"),
    COMMA         (BINARY,","),
    DOT           (BINARY,"."),
    ELLIPSIS      (BINARY,"..."),
    AT            (BINARY,"@"),
    COLONCOLON    (BINARY,"::"),

    // Operators
    ASSIGN         (BINARY, "="),
    GT             (BINARY, ">"),
    LT             (BINARY, "<"),
    BANG           (BINARY, "!"),
    TILDE          (BINARY, "~"),
    QUESTION       (BINARY, "?"),
    COLON          (BINARY, ":"),
    ARROW          (BINARY, "->"),
    EQUAL          (BINARY, "=="),
    LE             (BINARY, "<="),
    GE             (BINARY, ">="),
    NOTEQUAL       (BINARY, "!="),
    AND            (BINARY, "&&"),
    OR             (BINARY, "||"),
    INC            (BINARY, "++"),
    DEC            (BINARY, "--"),
    ADD            (BINARY, "+"),
    SUB            (BINARY, "-"),
    MUL            (BINARY, "*"),
    DIV            (BINARY, "/"),
    BITAND         (BINARY, "&"),
    BITOR          (BINARY, "|"),
    CARET          (BINARY, "^"),
    MOD            (BINARY, "%"),
    LSHIFT         (BINARY, "<<"),
    RSHIFT         (BINARY, ">>"),
    URSHIFT        (BINARY, ">>>"),
    ADD_ASSIGN     (BINARY, "+="),
    SUB_ASSIGN     (BINARY, "-="),
    MUL_ASSIGN     (BINARY, "*="),
    DIV_ASSIGN     (BINARY, "/="),
    AND_ASSIGN     (BINARY, "&="),
    OR_ASSIGN      (BINARY, "|="),
    XOR_ASSIGN     (BINARY, "^="),
    MOD_ASSIGN     (BINARY, "%="),
    LSHIFT_ASSIGN  (BINARY, "<<="),
    RSHIFT_ASSIGN  (BINARY, ">>="),
    URSHIFT_ASSIGN (BINARY, ">>>="),
    IDENT          (LITERAL,      "");

    //@formatter:on

    /**
     * Next token kind in token lookup table.
     */
    private TokenType next;

    /**
     * Classification of token.
     */
    private final TokenKind kind;

    /**
     * Printable name of token.
     */
    private final String name;

    /**
     * Operator precedence.
     */
    private final int precedence;

    /**
     * Left associativity
     */
    private final boolean isLeftAssociative;

    /**
     * ECMAScript version defining the token.
     */
    private final int version;

    /**
     * Cache values to avoid cloning.
     */
    private static final TokenType[] tokenValues;

    private static final Map<String, TokenType> map;

    TokenType(final TokenKind kind, final String name) {
        this(kind, name, 0, false);
    }

    TokenType(final TokenKind kind, final String name, final int precedence, final boolean isLeftAssociative) {
        this(kind, name, precedence, isLeftAssociative, 5);
    }

    TokenType(final TokenKind kind, final String name, final int precedence, final boolean isLeftAssociative, final int version) {
        next = null;
        this.kind = kind;
        this.name = name;
        this.precedence = precedence;
        this.isLeftAssociative = isLeftAssociative;
        this.version = version;
    }

    public String getName() {
        return name;
    }

    public String getNameOrType() {
        return name == null ? super.name().toLowerCase(Locale.ENGLISH) : name;
    }

    public TokenType getNext() {
        return next;
    }

    void setNext(final TokenType next) {
        this.next = next;
    }

    public TokenKind getKind() {
        return kind;
    }

    public int getPrecedence() {
        return precedence;
    }

    public boolean isLeftAssociative() {
        return isLeftAssociative;
    }

    public int getVersion() {
        return version;
    }

    public static boolean startsWith(final String index) {
        Set<String> collect = map.keySet().stream().filter(a -> a.startsWith(index)).collect(Collectors.toSet());
        return collect.size()>0;
    }

    static TokenType[] getValues() {
        return tokenValues;
    }

    @Override
    public String toString() {
        return getNameOrType();
    }

    public static TokenType getType(String index) {
        return map.get(index);
    }

    static {
        tokenValues = TokenType.values();
        map = new HashMap();
        for (TokenType c : TokenType.values()) {
            map.put(c.name, c);
        }
    }

}
