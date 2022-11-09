package com.deep.framework.ast.declaration;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.expression.Name;
import com.deep.framework.ast.lexer.TokenType;
import com.deep.framework.ast.type.Type;
import lombok.Data;

import java.util.List;

@Data
public class FieldDeclaration extends Declaration {
    private List<TokenType> modifiers;
    public Type type;
    public Name name;

    public FieldDeclaration(Node prarent) {
        super(prarent);
    }
}
