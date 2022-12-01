
package com.deep.framework.ast.lexer;


import com.deep.framework.ast.Node;
import com.deep.framework.ast.expression.Name;
import com.deep.framework.ast.literal.Literal;

import javax.lang.model.SourceVersion;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.stream.Collectors;

import static com.deep.framework.ast.lexer.TokenKind.*;

public enum TokenType {
    //@formatter:off

    // Keywords
    ABSTRACT      (KEYWORD,"abstract"),
    ASSERT        (KEYWORD,"assert"),
    BOOLEAN       (PRIMITIVE,"boolean"),
    BREAK         (KEYWORD,"break"),
    BYTE          (PRIMITIVE,"byte"),
    CASE          (KEYWORD,"case"),
    CATCH         (KEYWORD,"catch"),
    CHAR          (PRIMITIVE,"char"),
    CLASS         (KEYWORD,"class"),
    CONST         (KEYWORD,"const"),
    CONTINUE      (KEYWORD,"continue"),
    DEFAULT       (KEYWORD,"default"),
    DO            (KEYWORD,"do"),
    DOUBLE        (PRIMITIVE,"double"),
    ELSE          (KEYWORD,"else"),
    ENUM          (KEYWORD,"enum"),
    EXTENDS       (KEYWORD,"extends"),
    FINAL         (KEYWORD,"final"),
    FINALLY       (KEYWORD,"finally"),
    FLOAT         (PRIMITIVE,"float"),
    FOR           (KEYWORD,"for"),
    IF            (KEYWORD,"if"),
    GOTO          (KEYWORD,"goto"),
    IMPLEMENTS    (KEYWORD,"implements"),
    IMPORT        (KEYWORD,"import"),
    INSTANCEOF    (KEYWORD,"instanceof"),
    INT           (PRIMITIVE,"int"),
    INTERFACE     (KEYWORD,"interface"),
    LONG          (PRIMITIVE,"long"),
    NATIVE        (KEYWORD,"native"),
    NEW           (KEYWORD,"new"),
    PACKAGE       (KEYWORD,"package"),
    PRIVATE       (KEYWORD,"private"),
    PROTECTED     (KEYWORD,"protected"),
    PUBLIC        (KEYWORD,"public"),
    RETURN        (KEYWORD,"return"),
    SHORT         (PRIMITIVE,"short"),
    STATIC        (KEYWORD,"static"),
    STRICTFP      (KEYWORD,"strictfp"),
    SUPER         (IDENT,"super"),
    SWITCH        (KEYWORD,"switch"),
    SYNCHRONIZED  (KEYWORD,"synchronized"),
    THIS          (IDENT,"this"),
    THROW         (KEYWORD,"throw"),
    THROWS        (KEYWORD,"throws"),
    TRANSIENT     (KEYWORD,"transient"),
    TRY           (KEYWORD,"try"),
    VOID          (KEYWORD,"void"),
    VOLATILE      (KEYWORD,"volatile"),
    WHILE         (KEYWORD,"while"),
    UNDER_SCORE   (KEYWORD,"_"),

    // Separators
    LPAREN        (BRACKET,"("),
    RPAREN        (BRACKET,")"),
    LBRACE        (BRACKET,"{"),
    RBRACE        (BRACKET,"}"),
    LBRACK        (BRACKET,"["),
    RBRACK        (BRACKET,"]"),
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
    IDENTIFIER     (IDENT,      "");

    //@formatter:on

    /**
     * Classification of token.
     */
    private final TokenKind kind;

    public Node getToken() {
        if (List.of(IDENT, PRIMITIVE).contains(this.kind)) return new Name(this);
        return new Node(this);
    }

    /**
     * Printable name of token.
     */
    private final String name;

    private static final Map<String, TokenType> map = Arrays.stream(TokenType.values()).collect(Collectors.toMap(TokenType::getName, a -> a));

    TokenType(final TokenKind kind, final String name) {
        this.kind = kind;
        this.name = name;
    }

    public String getName() {
        return name;
    }

    public TokenKind getKind() {
        return kind;
    }

    public String getNameOrType() {
        return name == null ? super.name().toLowerCase(Locale.ENGLISH) : name;
    }

    public static boolean contains(final String name) {
        return map.containsKey(name);
    }

    public static boolean startsWith(final String name) {
        return map.keySet().stream().anyMatch(a -> a.startsWith(name));
    }

    public static TokenType getType(String name) {
        return map.get(name);
    }

    public static boolean isNumber(String value){
        return Literal.isNumber(value);
    }

    public static boolean isIdentifier(CharSequence name) {
        return SourceVersion.isIdentifier(name);
    }

    public static boolean isWhitespace(String name) {
        for (char a : name.toCharArray()) {
            if (!Character.isWhitespace(a)) {
                return false;
            }
        }
        return true;
    }

    @Override
    public String toString() {
        return getNameOrType();
    }

}